import argparse
import csv
import time
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HISTORY = 12
HORIZONS = (3, 6, 12)
EPOCHS = 5
BATCH_SIZE = 16
HIDDEN_SIZE = 32
SEED = 42


class SharedSensorLSTM(nn.Module):
    """One shared LSTM processes every sensor independently."""

    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, len(HORIZONS))

    def forward(self, x):
        # x: (batch, history, sensors, 2)
        batch, history, sensors, features = x.shape

        # Apply identical LSTM weights to every sensor.
        x = x.permute(0, 2, 1, 3).reshape(
            batch * sensors, history, features
        )
        _, (hidden, _) = self.lstm(x)
        prediction = self.head(hidden[-1])

        return prediction.reshape(batch, sensors, -1).permute(0, 2, 1)


def masked_mae(prediction, target, mask):
    errors = torch.abs(prediction - target)
    weights = mask.float()
    return (errors * weights).sum() / weights.sum().clamp_min(1.0)


def metrics(pred, truth, mask):
    """Calculate metrics in original speed units."""
    results = []

    for j, horizon in enumerate(HORIZONS):
        valid = mask[:, j, :]
        p = pred[:, j, :][valid]
        y = truth[:, j, :][valid]

        if y.numel() == 0:
            results.append({
                "horizon_minutes": horizon * 5,
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "MAPE_percent": float("nan"),
                "support": 0,
            })
            continue

        error = p - y
        positive = y > 0
        mape = (
            (torch.abs(error[positive] / y[positive]).mean() * 100).item()
            if positive.any()
            else float("nan")
        )

        results.append({
            "horizon_minutes": horizon * 5,
            "MAE": torch.abs(error).mean().item(),
            "RMSE": torch.sqrt((error ** 2).mean()).item(),
            "MAPE_percent": mape,
            "support": int(y.numel()),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_path = root / "data" / "metr-la.h5"
    output_dir = root / "baseline_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"METR-LA file not found: {data_path}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print(f"Dataset: {data_path}")
    print(f"Device: {device}")

    with h5py.File(data_path, "r") as f:
        raw = f["df"]["block0_values"][:].astype(np.float32)

    n, sensors = raw.shape
    b1, b2 = int(0.7 * n), int(0.8 * n)
    observed = raw != 0

    # Fit scaling statistics on observed training readings only.
    train_observed = observed[:b1]
    counts = np.maximum(train_observed.sum(axis=0), 1)

    mean = (
        (raw[:b1] * train_observed).sum(axis=0) / counts
    ).astype(np.float32)

    variance = (
        (((raw[:b1] - mean) ** 2) * train_observed).sum(axis=0)
        / counts
    )
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0

    # Missing readings become zero after scaling, with a missingness flag.
    scaled = np.where(
        observed, (raw - mean) / std, 0.0
    ).astype(np.float32)

    inputs = np.stack(
        [scaled, observed.astype(np.float32)], axis=-1
    )

    # Load the same reviewed issue times used by all other models.
    def load_issue_times(filename):
        path = root / "split_ids" / filename
        rows = np.genfromtxt(
            path, delimiter=",", names=True,
            dtype=None, encoding="utf-8"
        )
        return np.atleast_1d(rows)["issue_row"].astype(int).tolist()

    train_times = load_issue_times("train_ids.csv")
    val_times = load_issue_times("validation_ids.csv")
    test_times = load_issue_times("test_ids.csv")

    assert len(train_times) == 3995
    assert len(val_times) == 285
    assert len(test_times) == 571

    print(
        f"Loaded official issue times: "
        f"train={len(train_times)}, validation={len(val_times)}, "
        f"test={len(test_times)}"
    )

    def make_arrays(times):
        x_list, y_list, mask_list = [], [], []

        for t in times:
            x_list.append(inputs[t - HISTORY + 1:t + 1])

            target_indices = [t + h for h in HORIZONS]
            y_list.append(scaled[target_indices])
            mask_list.append(observed[target_indices])  

        return (
            np.asarray(x_list, dtype=np.float32),
            np.asarray(y_list, dtype=np.float32),
            np.asarray(mask_list, dtype=bool),
        )

    x_train, y_train, m_train = make_arrays(train_times)
    x_val, y_val, m_val = make_arrays(val_times)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(m_train),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    model = SharedSensorLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training for at most {args.epochs} epochs.")
    print("Validation only; test data will not be evaluated.")

    best_val = float("inf")
    best_path = root / "shared_lstm_best.pt"
    log_path = output_dir / "shared_lstm_validation.csv"
    start = time.perf_counter()
    log_rows = []

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    m_val_t = torch.from_numpy(m_val).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        batches = 0

        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mae(pred, yb, mb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = masked_mae(val_pred, y_val_t, m_val_t).item()

        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train masked MAE (scaled): {loss_sum / batches:.5f} | "
            f"validation masked MAE (scaled): {val_loss:.5f} | "
            f"epoch time: {elapsed:.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_masked_mae_scaled": loss_sum / batches,
            "validation_masked_mae_scaled": val_loss,
            "epoch_seconds": elapsed,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "best_validation_mae_scaled": best_val,
                "epoch": epoch,
                "parameters": parameter_count,
            }, best_path)

    total_seconds = time.perf_counter() - start

    # Restore the best validation checkpoint before reporting metrics.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x_val_t).cpu().numpy()

    # Convert forecasts back to original speed units.
    pred_original = pred_scaled * std[None, None, :] + mean[None, None, :]
    truth_original = y_val * std[None, None, :] + mean[None, None, :]

    results = metrics(
        torch.from_numpy(pred_original),
        torch.from_numpy(truth_original),
        torch.from_numpy(m_val),
    )

    metrics_path = output_dir / "shared_lstm_validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "horizon_minutes", "MAE", "RMSE", "MAPE_percent", "support"
        ])
        writer.writeheader()
        writer.writerows(results)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_masked_mae_scaled",
            "validation_masked_mae_scaled", "epoch_seconds"
        ])
        writer.writeheader()
        writer.writerows(log_rows)

    print("\nValidation metrics (original speed units):")
    for row in results:
        print(
            f"{row['horizon_minutes']} min | "
            f"MAE={row['MAE']:.4f} | RMSE={row['RMSE']:.4f} | "
            f"MAPE={row['MAPE_percent']:.4f}% | "
            f"support={row['support']}"
        )

    print(f"\nTotal training time: {total_seconds:.1f}s")
    print(f"Best model: {best_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Epoch log: {log_path}")


if __name__ == "__main__":
    main()


