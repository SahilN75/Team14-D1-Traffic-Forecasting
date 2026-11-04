
import pandas as pd
from pathlib import Path

DATA = Path("data/metr-la.h5")
OUT = Path("split_ids")
OUT.mkdir(exist_ok=True)

df = pd.read_hdf(DATA)
N = len(df)

train_end = int(0.70 * N)
val_end = int(0.80 * N)

# Each issue time t predicts t+3, t+6 and t+12.
# All target rows must remain within the same partition.
partitions = {
    "train": (0, train_end, 6),
    "validation": (train_end, val_end, 12),
    "test": (val_end, N, 12),
}

for name, (start, end, stride) in partitions.items():
    # The 12-step history must exist.
    # Targets must stay inside the partition.
    first_t = max(start, 12)
    last_t = end - 1 - 12

    eligible = list(range(first_t, last_t + 1))
    issue_ids = eligible[::stride]

    result = pd.DataFrame({
        "issue_row": issue_ids,
        "issue_timestamp": [
            str(df.index[t]) for t in issue_ids
        ],
        "target_15min_row": [t + 3 for t in issue_ids],
        "target_30min_row": [t + 6 for t in issue_ids],
        "target_60min_row": [t + 12 for t in issue_ids],
    })

    result.to_csv(OUT / f"{name}_ids.csv", index=False)

    print(
        f"{name}: eligible={len(eligible)}, "
        f"sampled={len(issue_ids)}, "
        f"saved={OUT / f'{name}_ids.csv'}"
    )

print("\nSplit-ID generation complete.")
print("Dataset rows:", N)
print("Partition boundaries:", train_end, val_end)
