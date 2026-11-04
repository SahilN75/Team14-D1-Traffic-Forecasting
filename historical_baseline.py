
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data/metr-la.h5")
SPLITS = Path("split_ids")
OUT = Path("baseline_results")
OUT.mkdir(exist_ok=True)

start_time = time.perf_counter()

df = pd.read_hdf(DATA)
values = df.to_numpy(dtype=np.float64)
N, sensors = values.shape

train_end = int(0.70 * N)

# Zero denotes a missing observation.
observed = values != 0
train_values = values[:train_end]
train_observed = observed[:train_end]

# Per-sensor training means from observed training values only.
counts = train_observed.sum(axis=0)
sums = np.where(train_observed, train_values, 0).sum(axis=0)
sensor_means = np.divide(
    sums, counts,
    out=np.zeros(sensors),
    where=counts > 0
)

# Historical average by weekday and 5-minute target slot.
train_index = df.index[:train_end]
slot_keys = (
    train_index.dayofweek * 288
    + train_index.hour * 12
    + train_index.minute // 5
)

history = {}
for key in np.unique(slot_keys):
    mask = slot_keys == key
    vals = train_values[mask]
    obs = train_observed[mask]

    cnt = obs.sum(axis=0)
    sm = np.where(obs, vals, 0).sum(axis=0)

    history[int(key)] = np.divide(
        sm, cnt,
        out=sensor_means.copy(),
        where=cnt > 0
    )

def slot_key(ts):
    return int(
        ts.dayofweek * 288
        + ts.hour * 12
        + ts.minute // 5
    )

# Week 2: validation only. Do not score test.
ids = pd.read_csv(SPLITS / "validation_ids.csv")
issue_rows = ids["issue_row"].to_numpy(dtype=int)

rows = []

for horizon, offset in [(15, 3), (30, 6), (60, 12)]:
    errors = []
    abs_pct = []
    n_scored = 0

    for t in issue_rows:
        target = t + offset

        # Every forecast target must remain in validation.
        if target >= len(df) or target < train_end:
            continue

        truth = values[target]
        valid = observed[target] & (truth > 0)

        if not valid.any():
            continue

        # Use the target timestamp, not the issue timestamp.
        prediction = history.get(
            slot_key(df.index[target]),
            sensor_means
        )

        err = prediction[valid] - truth[valid]
        errors.append(err)
        abs_pct.append(np.abs(err) / truth[valid])
        n_scored += int(valid.sum())

    if errors:
        e = np.concatenate(errors)
        p = np.concatenate(abs_pct)
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e ** 2))
        mape = np.mean(p) * 100
    else:
        mae = rmse = mape = float("nan")

    rows.append({
        "split": "validation",
        "horizon_minutes": horizon,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "scored_sensor_targets": n_scored,
    })

result = pd.DataFrame(rows)
result.to_csv(
    OUT / "historical_average_results.csv",
    index=False
)

elapsed = time.perf_counter() - start_time

print(result.to_string(index=False))
print(f"\nElapsed time: {elapsed:.2f} seconds")
print(f"Saved: {OUT / 'historical_average_results.csv'}")
