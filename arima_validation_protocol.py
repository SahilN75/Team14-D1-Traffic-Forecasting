
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA = Path("data/metr-la.h5")
SPLITS = Path("split_ids")
OUT = Path("baseline_results")
OUT.mkdir(parents=True, exist_ok=True)

ORDER = (2, 0, 1)
K = 20

HORIZONS = [
    (15, "target_15min_row", 3),
    (30, "target_30min_row", 6),
    (60, "target_60min_row", 12),
]

wall_start = time.perf_counter()


# --------------------------------------------------
# Load data and validate chronological split
# --------------------------------------------------
df = pd.read_hdf(DATA)
X = df.to_numpy(dtype=np.float64)
N, S = X.shape

train_end = int(0.70 * N)
val_end = int(0.80 * N)

ids = pd.read_csv(SPLITS / "validation_ids.csv")

required_columns = [
    "issue_row",
    *[col for _, col, _ in HORIZONS],
]
missing_columns = [c for c in required_columns if c not in ids.columns]
if missing_columns:
    raise ValueError(f"Missing split columns: {missing_columns}")

issues = ids["issue_row"].to_numpy(dtype=int)
target_rows = {
    horizon: ids[column].to_numpy(dtype=int)
    for horizon, column, _ in HORIZONS
}

if len(issues) == 0:
    raise ValueError("No validation issue rows found.")

if np.any(np.diff(issues) <= 0):
    raise ValueError("Validation issue rows must be strictly increasing.")

if np.any(issues < train_end) or np.any(issues >= val_end):
    raise ValueError("An issue row lies outside the validation partition.")

for horizon, column, steps in HORIZONS:
    targets = target_rows[horizon]

    if np.any(targets < train_end) or np.any(targets >= val_end):
        raise ValueError(f"{column}: target outside validation partition.")

    if not np.array_equal(targets, issues + steps):
        raise ValueError(
            f"{column}: targets do not equal issue_row + {steps}."
        )


# --------------------------------------------------
# Missing values and training-only means
# --------------------------------------------------
# Zero denotes a missing speed reading.
observed = X != 0
train_obs = observed[:train_end]

counts = train_obs.sum(axis=0)
if np.any(counts == 0):
    raise ValueError("At least one sensor has no observed training values.")

means = (
    np.where(train_obs, X[:train_end], 0).sum(axis=0)
    / counts
)

# Missing input values are filled using training-only sensor means.
Y = np.where(observed, X, means[np.newaxis, :])


# --------------------------------------------------
# Historical-average fallback
# Mean by weekday and 5-minute time slot, using
# observed training speeds only.
# --------------------------------------------------
train_index = df.index[:train_end]

keys = (
    train_index.dayofweek * 288
    + train_index.hour * 12
    + train_index.minute // 5
).to_numpy()

history = {}

for key in np.unique(keys):
    mask = keys == key
    vals = X[:train_end][mask]
    obs = observed[:train_end][mask]

    slot_counts = obs.sum(axis=0)
    slot_sums = np.where(obs, vals, 0).sum(axis=0)

    history[int(key)] = np.divide(
        slot_sums,
        slot_counts,
        out=means.copy(),
        where=slot_counts > 0,
    )


def slot_key(timestamp):
    return int(
        timestamp.dayofweek * 288
        + timestamp.hour * 12
        + timestamp.minute // 5
    )


# --------------------------------------------------
# Results and scoring helpers
# --------------------------------------------------
errors = {h: [] for h, _, _ in HORIZONS}
pct_errors = {h: [] for h, _, _ in HORIZONS}
scored_counts = {h: 0 for h, _, _ in HORIZONS}

fit_seconds = 0.0
roll_forecast_seconds = 0.0

fit_failures = []
convergence_warnings = []
fallback_sensors = []

sensors_run = min(K, S)


def score_fallback_sensor(sensor):
    """Score failed sensors using the historical average."""
    fallback_sensors.append(sensor)

    for horizon, _, _ in HORIZONS:
        targets = target_rows[horizon]

        for target in targets:
            # Exclude missing and non-positive ground-truth speeds.
            if observed[target, sensor] and X[target, sensor] > 0:
                key = slot_key(df.index[target])
                prediction = history[key][sensor]

                error = float(prediction - X[target, sensor])
                errors[horizon].append(error)
                pct_errors[horizon].append(
                    abs(error) / X[target, sensor]
                )
                scored_counts[horizon] += 1


# --------------------------------------------------
# Fit ARIMA once per sensor and roll forward
# chronologically using observations available
# through each validation issue time.
# --------------------------------------------------
for j in range(sensors_run):
    print(f"Sensor {j + 1}/{sensors_run}", flush=True)

    fit_start = time.perf_counter()

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            fitted = ARIMA(
                Y[:train_end, j],
                order=ORDER,
                trend="c",
            ).fit()

        fit_seconds += time.perf_counter() - fit_start

        has_convergence_warning = any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught
        )

        if has_convergence_warning:
            convergence_warnings.append(j)
            fit_failures.append(
                (j, "ConvergenceWarning; historical-average fallback used")
            )
            score_fallback_sensor(j)
            continue

    except Exception as exc:
        fit_seconds += time.perf_counter() - fit_start
        fit_failures.append((j, repr(exc)))
        score_fallback_sensor(j)
        continue

    # The fitted model initially knows training observations only.
    last_seen = train_end - 1

    for issue_index, issue in enumerate(issues):
        forecast_start = time.perf_counter()

        # Advance the state using observations up to and including
        # this issue time. No future target observations are appended.
        if issue > last_seen:
            fitted = fitted.append(
                Y[last_seen + 1 : issue + 1, j],
                refit=False,
            )
            last_seen = issue

        forecasts = np.asarray(
            fitted.get_forecast(steps=12).predicted_mean,
            dtype=np.float64,
        )

        roll_forecast_seconds += time.perf_counter() - forecast_start

        for horizon, _, steps in HORIZONS:
            target = int(target_rows[horizon][issue_index])

            # Only observed, positive ground-truth speeds are scored.
            if observed[target, j] and X[target, j] > 0:
                prediction = float(forecasts[steps - 1])
                error = prediction - X[target, j]

                errors[horizon].append(float(error))
                pct_errors[horizon].append(
                    abs(float(error)) / X[target, j]
                )
                scored_counts[horizon] += 1


# --------------------------------------------------
# Metrics
# --------------------------------------------------
rows = []

for horizon, _, _ in HORIZONS:
    e = np.asarray(errors[horizon], dtype=np.float64)
    p = np.asarray(pct_errors[horizon], dtype=np.float64)

    rows.append({
        "sensors_attempted": sensors_run,
        "horizon_minutes": horizon,
        "MAE": float(np.mean(np.abs(e))) if len(e) else np.nan,
        "RMSE": float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
        "MAPE_percent": float(np.mean(p) * 100) if len(p) else np.nan,
        "scored_sensor_targets": int(scored_counts[horizon]),
    })

result_df = pd.DataFrame(rows)

results_path = OUT / "arima_validation_results.csv"
result_df.to_csv(results_path, index=False)


# --------------------------------------------------
# Runtime summary
# --------------------------------------------------
elapsed = time.perf_counter() - wall_start
scale = S / sensors_run

print("\nARIMA(2,0,1) validation results — first 20 sensors")
print(result_df.to_string(index=False))

print(f"\nFit elapsed: {fit_seconds:.2f} s")
print(f"State-roll/forecast elapsed: {roll_forecast_seconds:.2f} s")
print(f"Total elapsed: {elapsed:.2f} s")

print(
    f"Projected 207-sensor fit: "
    f"{fit_seconds * scale / 60:.2f} min"
)
print(
    f"Projected 207-sensor state-roll/forecast: "
    f"{roll_forecast_seconds * scale / 60:.2f} min"
)

print("\nFit failures / fallback reasons:")
for sensor, reason in fit_failures:
    print(f"  Sensor {sensor}: {reason}")

print("Convergence-warning sensor indices:", convergence_warnings)
print("Historical-average fallback sensor indices:", sorted(set(fallback_sensors)))
print("Historical-average fallback sensor count:", len(set(fallback_sensors)))
print("Saved:", results_path)
print("Test partition was not evaluated.")
