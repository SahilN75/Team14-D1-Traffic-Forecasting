"""
Team 14 / Card D1 - remaining full validation experiments.

Run from the DCRNN_Official repository root:
    python d1_remaining_experiments.py

Optional:
    python d1_remaining_experiments.py --device cuda --threads 10 --max-epochs 30
    python d1_remaining_experiments.py --device cpu --threads 10 --max-minutes 30

Uses only the reviewed train/validation issue IDs. Test issue IDs and test labels
are deliberately not loaded. Existing gcngru_best.pt and shared_lstm_best.pt
are read-only inputs; new checkpoints/logs are written under experiments_d1/.
"""
import argparse
import csv
import json
import os
import pickle
import random
import time
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HISTORY = 12
HORIZONS = (3, 6, 12)  # 15, 30, 60 minutes
SEED = 22
BATCH_SIZE = 32
PATIENCE = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_issue_times(path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(rows)["issue_row"].astype(int)


def masked_mae(pred, target, mask):
    weights = mask.float()
    return ((pred - target).abs() * weights).sum() / weights.sum().clamp_min(1.0)


def build_metrics(pred_scaled, truth_scaled, mask, mean, std):
    pred = pred_scaled * std[None, None, :] + mean[None, None, :]
    truth = truth_scaled * std[None, None, :] + mean[None, None, :]
    rows = []
    for j, horizon in enumerate(HORIZONS):
        valid = mask[:, j, :]
        p = pred[:, j, :][valid]
        y = truth[:, j, :][valid]
        if y.numel() == 0:
            rows.append({
                "horizon_minutes": horizon * 5,
                "MAE": float("nan"), "RMSE": float("nan"),
                "MAPE_percent": float("nan"), "support": 0
            })
            continue
        err = p - y
        positive = y > 0
        mape = (torch.abs(err[positive] / y[positive]).mean() * 100).item() if positive.any() else float("nan")
        rows.append({
            "horizon_minutes": horizon * 5,
            "MAE": torch.abs(err).mean().item(),
            "RMSE": torch.sqrt((err ** 2).mean()).item(),
            "MAPE_percent": mape,
            "support": int(y.numel())
        })
    return rows


class SensorTimeCNN(nn.Module):
    """2-D convolution over the sensor-by-time input grid."""
    def __init__(self, sensors):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, sensors)),
        )
        self.head = nn.Conv1d(16, 3, kernel_size=1)

    def forward(self, x):
        z = x.permute(0, 3, 1, 2)  # B, features, time, sensors
        z = self.net(z).squeeze(2)
        return self.head(z)


class GraphAttention(nn.Module):
    """Single-head graph-masked attention over the sensor nodes."""
    def __init__(self, adjacency_mask):
        super().__init__()
        self.register_buffer("A_mask", adjacency_mask.bool())
        self.q = nn.Linear(2, 16)
        self.k = nn.Linear(2, 16)
        self.v = nn.Linear(2, 16)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        z = x[:, -1]  # B, sensors, 2
        q, k, v = self.q(z), self.k(z), self.v(z)
        scores = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        scores = scores.masked_fill(~self.A_mask.unsqueeze(0), -1e4)
        weights = scores.softmax(dim=-1)
        return self.head(torch.matmul(weights, v)).permute(0, 2, 1)


class AttentionAblation(nn.Module):
    """Attention ablation: replace learned attention with uniform graph-neighbour averaging."""
    def __init__(self, adjacency_mask):
        super().__init__()
        a = adjacency_mask.float()
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("A_mean", a)
        self.value = nn.Linear(2, 16)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        z = self.value(x[:, -1])
        aggregated = torch.matmul(self.A_mean.unsqueeze(0), z)
        return self.head(aggregated).permute(0, 2, 1)


class GCNGRU(nn.Module):
    """Architecture matching gcngru_timed_run.py checkpoint key names."""
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


class SharedSensorLSTM(nn.Module):
    """Architecture matching shared_lstm.py checkpoint key names."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, len(HORIZONS))

    def forward(self, x):
        b, t, n, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, (hidden, _) = self.lstm(z)
        return self.head(hidden[-1]).reshape(b, n, -1).permute(0, 2, 1)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    outdir = root / "experiments_d1_cpu"
    outdir.mkdir(exist_ok=True)

    data_path = root / "data" / "metr-la.h5"
    graph_path = root / "data" / "sensor_graph" / "adj_mx.pkl"
    train_path = root / "split_ids" / "train_ids.csv"
    val_path = root / "split_ids" / "validation_ids.csv"
    for p in (data_path, graph_path, train_path, val_path,
              root / "gcngru_best.pt", root / "shared_lstm_best.pt"):
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    print("Team 14 D1 remaining experiments")
    print(f"Device: {device}; torch threads: {torch.get_num_threads()}; workers: 0")
    print("Test data: NOT LOADED; only train and validation issue IDs are used.")

    with h5py.File(data_path, "r") as f:
        raw = f["df"]["block0_values"][:].astype(np.float32)
    n_rows, sensors = raw.shape
    observed = raw != 0
    train_observed = observed[:int(0.7 * n_rows)]
    counts = np.maximum(train_observed.sum(axis=0), 1)
    mean = ((raw[:int(0.7 * n_rows)] * train_observed).sum(axis=0) / counts).astype(np.float32)
    variance = (
        (((raw[:int(0.7 * n_rows)] - mean) ** 2) * train_observed).sum(axis=0) / counts
    )
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0
    scaled = np.where(observed, (raw - mean) / std, 0.0).astype(np.float32)
    inputs = np.stack([scaled, observed.astype(np.float32)], axis=-1)

    train_times = load_issue_times(train_path)
    val_times = load_issue_times(val_path)
    if len(train_times) != 3995 or len(val_times) != 285:
        raise ValueError(
            f"Unexpected reviewed split sizes: train={len(train_times)}, val={len(val_times)}"
        )

    def make_arrays(times):
        x = np.stack([inputs[t - HISTORY + 1:t + 1] for t in times]).astype(np.float32)
        target_rows = np.stack([[t + h for h in HORIZONS] for t in times])
        y = scaled[target_rows].astype(np.float32)
        m = observed[target_rows].astype(bool)
        return x, y, m

    xtr, ytr, mtr = make_arrays(train_times)
    xva, yva, mva = make_arrays(val_times)
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr)
    mtr_t = torch.from_numpy(mtr)
    xva_t = torch.from_numpy(xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)
    mva_t = torch.from_numpy(mva).to(device)

    # Match the row-normalised graph used by the existing GCN+GRU fit.
    ids, _, adjacency_raw = pickle.load(open(graph_path, "rb"), encoding="latin1")
    adjacency_raw = np.asarray(adjacency_raw, dtype=np.float32)
    adjacency = adjacency_raw / np.maximum(adjacency_raw.sum(1, keepdims=True), 1e-8)
    adjacency_t = torch.from_numpy(adjacency).to(device)
    adjacency_mask = torch.from_numpy(adjacency_raw > 0).to(device)

    print(f"METR-LA: rows={n_rows}, sensors={sensors}")
    print(f"Reviewed issue IDs: train={len(train_times)}, validation={len(val_times)}")
    print(f"Input tensors: train={tuple(xtr.shape)}, validation={tuple(xva.shape)}")
    print(f"Output folder: {outdir}")

    train_ds = TensorDataset(xtr_t, ytr_t, torch.from_numpy(mtr))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed)
    )

    models = [
        ("cnn2d", SensorTimeCNN(sensors).to(device)),
        ("graph_attention", GraphAttention(adjacency_mask).to(device)),
        ("attention_ablation", AttentionAblation(adjacency_mask).to(device)),
    ]
    fit_summaries = []
    process = psutil.Process(os.getpid())

    for name, model in models:
        set_seed(args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if params > 250_000:
            raise RuntimeError(f"{name} has {params} parameters, exceeding 250,000.")
        best_val = float("inf")
        best_epoch = 0
        bad_epochs = 0
        updates = 0
        epoch_rows = []
        checkpoint_path = outdir / f"{name}_best.pt"
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        stop_reason = "epochs"
        print(f"\n=== FIT {name} | parameters={params:,} ===")

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            batches = 0
            permutation = torch.randperm(len(xtr_t))
            for begin in range(0, len(permutation), args.batch_size):
                indices = permutation[begin:begin + args.batch_size]
                xb = xtr_t[indices].to(device)
                yb = ytr_t[indices].to(device)
                mb = torch.from_numpy(mtr[indices.numpy()]).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                loss = masked_mae(prediction, yb, mb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
                updates += 1
                if time.perf_counter() - start_wall >= args.max_minutes * 60:
                    stop_reason = "clock"
                    break

            model.eval()
            with torch.no_grad():
                val_loss = masked_mae(model(xva_t), yva_t, mva_t).item()
            epoch_seconds = time.perf_counter() - epoch_start
            train_loss = total_loss / max(batches, 1)
            epoch_rows.append({
                "epoch": epoch, "train_masked_mae_scaled": train_loss,
                "validation_masked_mae_scaled": val_loss,
                "epoch_seconds": epoch_seconds, "updates_cumulative": updates
            })
            print(
                f"{name} epoch {epoch}/{args.max_epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"seconds={epoch_seconds:.1f} | updates={updates}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": name, "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_validation_mae_scaled": best_val,
                    "train_issue_count": len(train_times),
                    "validation_issue_count": len(val_times),
                    "parameters": params,
                }, checkpoint_path)
            else:
                bad_epochs += 1

            if stop_reason == "clock":
                break
            if bad_epochs >= args.patience:
                stop_reason = "early_stop"
                break

        elapsed = time.perf_counter() - start_wall
        cpu_seconds = time.process_time() - start_cpu
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            val_prediction = model(xva_t)
        metrics_rows = build_metrics(
            val_prediction.cpu(), yva_t.cpu(), mva_t.cpu(), mean, std
        )
        write_csv(
            outdir / f"{name}_epochs.csv", epoch_rows,
            ["epoch", "train_masked_mae_scaled", "validation_masked_mae_scaled",
             "epoch_seconds", "updates_cumulative"]
        )
        write_csv(
            outdir / f"{name}_validation_metrics.csv", metrics_rows,
            ["horizon_minutes", "MAE", "RMSE", "MAPE_percent", "support"]
        )
        try:
            peak_mem_mb = process.memory_info().peak_wset / (1024 ** 2)
        except AttributeError:
            peak_mem_mb = process.memory_info().rss / (1024 ** 2)

        summary = {
            "component": name, "seed": args.seed,
            "train_issue_count": len(train_times),
            "validation_issue_count": len(val_times),
            "epochs_completed": len(epoch_rows), "updates": updates,
            "stop_reason": stop_reason, "elapsed_s": round(elapsed, 2),
            "cpu_s": round(cpu_seconds, 2),
            "peak_mem_mb": round(peak_mem_mb, 2),
            "threads": torch.get_num_threads(), "workers": 0,
            "trainable_params": params, "best_epoch": best_epoch,
            "best_validation_masked_mae_scaled": best_val,
            "checkpoint": str(checkpoint_path),
            "metrics_file": str(outdir / f"{name}_validation_metrics.csv")
        }
        fit_summaries.append(summary)
        print("FIT SUMMARY:", json.dumps(summary, indent=2))
        print("VALIDATION METRICS:")
        for row in metrics_rows:
            print(
                f"  {row['horizon_minutes']} min: MAE={row['MAE']:.4f}, "
                f"RMSE={row['RMSE']:.4f}, MAPE={row['MAPE_percent']:.4f}%, "
                f"support={row['support']}"
            )

    # Validation-only 50/50 prediction fusion using the already trained checkpoints.
    print("\n=== VALIDATION-ONLY FUSION: existing GCN+GRU + existing shared LSTM ===")
    gcngru = GCNGRU(adjacency_t).to(device)
    gcngru_state = torch.load(root / "gcngru_best.pt", map_location=device, weights_only=False)
    gcngru.load_state_dict(gcngru_state)
    lstm = SharedSensorLSTM().to(device)
    lstm_obj = torch.load(root / "shared_lstm_best.pt", map_location=device, weights_only=False)
    lstm_state = lstm_obj["model_state_dict"] if isinstance(lstm_obj, dict) and "model_state_dict" in lstm_obj else lstm_obj
    lstm.load_state_dict(lstm_state)
    gcngru.eval()
    lstm.eval()
    fusion_start = time.perf_counter()
    with torch.no_grad():
        graph_prediction = gcngru(xva_t)
        lstm_prediction = lstm(xva_t)
        fusion_prediction = 0.5 * graph_prediction + 0.5 * lstm_prediction
    fusion_elapsed = time.perf_counter() - fusion_start
    fusion_metrics = build_metrics(
        fusion_prediction.cpu(), yva_t.cpu(), mva_t.cpu(), mean, std
    )
    write_csv(
        outdir / "fusion_gcn_gru_shared_lstm_validation_metrics.csv",
        fusion_metrics,
        ["horizon_minutes", "MAE", "RMSE", "MAPE_percent", "support"]
    )
    fusion_summary = {
        "component": "fusion", "method": "fixed 50/50 prediction average",
        "training_performed": False,
        "inputs": ["existing gcngru_best.pt", "existing shared_lstm_best.pt"],
        "validation_issue_count": len(val_times),
        "elapsed_s": round(fusion_elapsed, 4),
        "test_metrics_computed": False,
        "metrics_file": str(outdir / "fusion_gcn_gru_shared_lstm_validation_metrics.csv")
    }
    print("FUSION SUMMARY:", json.dumps(fusion_summary, indent=2))
    for row in fusion_metrics:
        print(
            f"  {row['horizon_minutes']} min: MAE={row['MAE']:.4f}, "
            f"RMSE={row['RMSE']:.4f}, MAPE={row['MAPE_percent']:.4f}%, "
            f"support={row['support']}"
        )

    with (outdir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "device": str(device), "seed": args.seed,
            "train_issue_count": len(train_times),
            "validation_issue_count": len(val_times),
            "test_data_loaded": False,
            "fit_summaries": fit_summaries,
            "fusion_summary": fusion_summary
        }, f, indent=2)

    print("\nDONE. New files are in experiments_d1/.")
    print("Test labels/metrics were not loaded or computed.")
    print("NOTE: fusion is fixed-weight inference-only fusion, not joint training.")


if __name__ == "__main__":
    main()
