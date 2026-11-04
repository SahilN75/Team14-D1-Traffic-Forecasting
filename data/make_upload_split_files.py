
import csv
import hashlib
import h5py
from pathlib import Path

BASE = Path(__file__).resolve().parent
H5_PATH = BASE / "metr-la.h5"

SPLIT_PATH = BASE / "T14_D1_metr-la.csv"
SUMMARY_PATH = BASE / "T14_D1_split_summary.csv"

HISTORY = 12
STRIDES = {"train": 6, "val": 12, "test": 12}

# Read dataset length safely.
with h5py.File(H5_PATH, "r") as h5:
    N = h5["df"]["block0_values"].shape[0]

b1 = int(0.70 * N)
b2 = int(0.80 * N)

# Issue times must belong to their partition, have 12 history
# observations available, and keep every target inside it.
bounds = {
    "train": (0, b1),
    "val": (b1, b2),
    "test": (b2, N),
}

parts = {}
for name, (lo, hi) in bounds.items():
    first = max(lo, HISTORY)
    stop = hi - HISTORY  # exclusive; last issue time is hi - 13
    parts[name] = list(range(first, stop, STRIDES[name]))

expected = {"train": 3995, "val": 285, "test": 571}
actual = {name: len(ids) for name, ids in parts.items()}
print("Dataset rows:", N)
print("Partition boundaries:", b1, b2)
print("Counts:", actual)

if actual != expected:
    raise ValueError(
        f"Counts do not match D4 expected counts: {expected}. "
        "No submission files were written."
    )

# Back up existing output files before replacing them.
for path in (SPLIT_PATH, SUMMARY_PATH):
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        if backup.exists():
            raise FileExistsError(
                f"Backup already exists: {backup}. "
                "Move it elsewhere before rerunning."
            )
        path.replace(backup)
        print("Backup created:", backup.name)

with SPLIT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        ["sample_id", "unit_id", "partition", "label", "reason"]
    )
    for partition in ("train", "val", "test"):
        for t in parts[partition]:
            writer.writerow([t, t, partition, "", ""])

sha256 = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()

with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        ["file", "partition", "n_samples", "n_units", "sha256"]
    )
    for partition in ("train", "val", "test", "excluded"):
        count = len(parts.get(partition, []))
        writer.writerow(
            [SPLIT_PATH.name, partition, count, count, sha256]
        )

print("\nCreated:")
print(SPLIT_PATH)
print(SUMMARY_PATH)
print("SHA-256:", sha256)
print("Backups have .bak extensions; keep them until verification.")
