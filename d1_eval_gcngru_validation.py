"""Validation-only original-unit metrics for the existing Team 14 GCN+GRU checkpoint.

Run from DCRNN_Official:
    python d1_eval_gcngru_validation.py

Only train and validation issue IDs are loaded. Test IDs/labels are never read.
The existing gcngru_best.pt is read-only.
"""
import csv
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

HISTORY = 12
HORIZONS = (3, 6, 12)


class GCNGRU(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.gc = nn.Linear(2, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        b, t, n, f = x.shape
        h = torch.relu(self.gc(torch.einsum("ij,btjf->btif", self.A, x)))
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        out, _ = self.gru(h)
        return self.head(out[:, -1]).reshape(b, n, 3).permute(0, 2, 1)


def load_issue_times(path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(rows)["issue_row"].astype(int)


def main():
    root = Path(".").resolve()
    data_path = root / "data" / "metr-la.h5"
    split_dir = root / "split_ids"
    graph_path = root / "data" / "sensor_graph" / "adj_mx.pkl"
    checkpoint_path = root / "gcngru_best.pt"
    out_dir = root / "experiments_d1"
    out_dir.mkdir(exist_ok=True)

    for p in (data_path, split_dir / "train_ids.csv",
              split_dir / "validation_ids.csv", graph_path, checkpoint_path):
        if not p.exists():
            raise FileNotFoundError(p)

    torch.set_num_threads(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; threads: {torch.get_num_threads()}")
    print("Evaluation scope: validation only; test labels are not loaded.")

    with h5py.File(data_path, "r") as f:
        raw = f["df"]["block0_values"][:].astype(np.float32)

    n_rows, sensors = raw.shape
    observed = raw != 0
    train_end = int(0.7 * n_rows)
    train_obs = observed[:train_end]
    counts = np.maximum(train_obs.sum(axis=0), 1)
    mean = ((raw[:train_end] * train_obs).sum(axis=0) / counts).astype(np.float32)
    variance = (
        (((raw[:train_end] - mean) ** 2) * train_obs).sum(axis=0) / counts
    )
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0

    scaled = np.where(observed, (raw - mean) / std, 0.0).astype(np.float32)
    inputs = np.stack([scaled, observed.astype(np.float32)], axis=-1)

    train_ids = load_issue_times(split_dir / "train_ids.csv")
    val_ids = load_issue_times(split_dir / "validation_ids.csv")
    if len(train_ids) != 3995 or len(val_ids) != 285:
        raise ValueError(f"Unexpected split sizes: train={len(train_ids)}, val={len(val_ids)}")

    x_val = np.stack([
        inputs[t - HISTORY + 1:t + 1] for t in val_ids
    ]).astype(np.float32)
    target_rows = np.stack([[t + h for h in HORIZONS] for t in val_ids])
    y_val = scaled[target_rows].astype(np.float32)
    mask_val = observed[target_rows].astype(bool)

    with open(graph_path, "rb") as f:
        _, _, adjacency_raw = pickle.load(f, encoding="latin1")
    adjacency_raw = np.asarray(adjacency_raw, dtype=np.float32)
    adjacency = adjacency_raw / np.maximum(adjacency_raw.sum(1, keepdims=True), 1e-8)

    model = GCNGRU(torch.from_numpy(adjacency).to(device)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    x_tensor = torch.from_numpy(x_val).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy()
    inference_seconds = time.perf_counter() - start

    pred_original = pred_scaled * std[None, None, :] + mean[None, None, :]
    truth_original = y_val * std[None, None, :] + mean[None, None, :]

    rows = []
    for j, horizon in enumerate(HORIZONS):
        valid = mask_val[:, j, :]
        p = pred_original[:, j, :][valid]
        y = truth_original[:, j, :][valid]
        error = p - y
        positive = y > 0
        mape = float(np.mean(np.abs(error[positive] / y[positive])) * 100) if positive.any() else float("nan")
        rows.append({
            "horizon_minutes": horizon * 5,
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE_percent": mape,
            "support": int(y.size),
        })

    output = out_dir / "gcngru_validation_metrics_original_units.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "horizon_minutes", "MAE", "RMSE", "MAPE_percent", "support"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Validation issue times: {len(val_ids)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Validation inference elapsed: {inference_seconds:.4f}s")
    print("\nGCN+GRU validation metrics (original speed units):")
    for row in rows:
        print(
            f"{row['horizon_minutes']} min | MAE={row['MAE']:.4f} | "
            f"RMSE={row['RMSE']:.4f} | MAPE={row['MAPE_percent']:.4f}% | "
            f"support={row['support']}"
        )
    print(f"\nSaved: {output}")
    print("Test metrics: NOT COMPUTED.")


if __name__ == "__main__":
    main()
