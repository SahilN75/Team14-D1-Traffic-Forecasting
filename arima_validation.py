
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

DATA = Path("data/metr-la.h5")
SPLITS = Path("split_ids")
OUT = Path("baseline_results")
OUT.mkdir(exist_ok=True)

ORDER = (2, 0, 1)
K = 20  # first 20 sensors, in file order
HORIZONS = [
    (15, "target_15min_row"),
    (30, "target_30min_row"),
    (60, "target_60min_row"),
]

start_all = time.perf_counter()

# Load the data and validation issue IDs.
df = pd.read_hdf(DATA)
X = df.to_numpy(dtype=np.float64)
N, S = X.shape
train_end = int(0.70 * N)
val_end = int(0.80 * N)

ids = pd.read_csv(SPLITS / "validation_ids.csv")

# Validate that all forecast targets remain in validation.
for _, col in HORIZONS:
    targets = ids[col].to_numpy(dtype=int)
    if np.any(targets < train_end) or np.any(targets >= val_end):
        raise ValueError(
            f"{col} contains targets outside the validation partition"
        )

# Missing readings are zeros. Impute using training-observed
# sensor means for the ARIMA input series.
observed = X != 0
train_obs = observed[:train_end]
counts = np.maximum(train_obs.sum(axis=0), 1)
means = np.where(train_obs, X[:train_end], 0).sum(axis=0) / counts
Y = np.where(observed, X, means)

fit_seconds = 0.0
forecast_seconds = 0.0
fit_failures = []
conv_warnings = []
results = []

for j in range(min(K, S)):
    print(f"Sensor {j + 1}/{min(K, S)}", flush=True)

    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ARIMA(
                Y[:train_end, j],
                order=ORDER,
                trend="c"
            ).fit()

        fit_seconds += time.perf_counter() - t0

        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            conv_warnings.append(j)

    except Exception as exc:
        fit_seconds += time.perf_counter() - t0
        fit_failures.append((j, str(exc)))
        print(f"  FIT FAILED: {exc}", flush=True)
        continue

    # Forecast from each issue time, using only data available
    # at or before that issue time. No future validation values
    # are used as inputs to a forecast.
    for minutes, target_col in HORIZONS:
        errors = []
        abs_pct = []
        target_rows = ids[target_col].to_numpy(dtype=int)
        issue_rows = ids["issue_row"].to_numpy(dtype=int)

        t1 = time.perf_counter()

        for issue, target in zip(issue_rows, target_rows):
            # Training history plus observations available by issue time.
            history = Y[:issue + 1, j]

            try:
                current_model = ARIMA(
                    history,
                    order=ORDER,
                    trend="c"
                ).fit()
                forecast = current_model.forecast(
                    steps=target - issue
                )[-1]
            except Exception:
                continue

            # Score observed validation targets only.
            truth = X[target, j]
            if observed[target, j] and truth > 0:
                err = float(forecast - truth)
                errors.append(err)
                abs_pct.append(abs(err) / truth)

        forecast_seconds += time.perf_counter() - t1

        if errors:
            e = np.asarray(errors)
            p = np.asarray(abs_pct)
            mae = np.mean(np.abs(e))
            rmse = np.sqrt(np.mean(e ** 2))
            mape = np.mean(p) * 100
            n_scored = len(e)
        else:
            mae = rmse = mape = float("nan")
            n_scored = 0

        results.append({
            "sensors_attempted": min(K, S),
            "horizon_minutes": minutes,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "scored_sensor_targets": n_scored,
        })

result = pd.DataFrame(results)
result.to_csv(OUT / "arima_validation_results.csv", index=False)

elapsed = time.perf_counter() - start_all
scale = S / min(K, S)

print("\nARIMA validation results (first 20 sensors)")
print(result.to_string(index=False))
print(f"\nFit time: {fit_seconds:.2f} seconds")
print(f"Repeated forecast/evaluation time: {forecast_seconds:.2f} seconds")
print(f"Total elapsed time: {elapsed:.2f} seconds")
print(f"Projected fit time for {S} sensors: {fit_seconds * scale / 60:.2f} minutes")
print(f"Projected forecast/evaluation time for {S} sensors: "
      f"{forecast_seconds * scale / 60:.2f} minutes")
print("Fit failures:", fit_failures)
print("Convergence warnings (sensor indices):", conv_warnings)
print("Saved:", OUT / "arima_validation_results.csv")
print("Test partition was not evaluated.")
