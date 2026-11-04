
"""
Team 14 / Card D1 checkpoint smoke test.

Run from repository root:
    python checkpoint_smoke.py --data data/metr-la.h5

This is a pipeline smoke test, NOT an experiment or a source of
reportable metrics.

It uses a tiny subset, trains each neural component for two updates,
validates, and performs unscored inference on test-shaped inputs.
No test targets are loaded.
"""

import argparse
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


def log(name, samples, start):
    elapsed = time.perf_counter() - start
    print(
        f"{name}: ran; samples={samples}; elapsed_s={elapsed:.2f}",
        flush=True
    )


class GraphGRU(nn.Module):
    def __init__(self, nodes, adj):
        super().__init__()
        self.register_buffer("A", adj)
        self.gru = nn.GRU(2, 16, batch_first=True)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        # x: B,T,N,2
        x = torch.einsum("ij,btjf->btif", self.A, x)
        b, t, n, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, h = self.gru(z)
        y = self.head(h[-1]).reshape(b, n, 3)
        return y.permute(0, 2, 1)


class SharedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 16, batch_first=True)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        b, t, n, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, (h, _) = self.lstm(z)
        return self.head(h[-1]).reshape(b, n, 3).permute(0, 2, 1)


class SensorTimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 207))
        )
        self.head = nn.Conv1d(8, 3, 1)

    def forward(self, x):
        z = x.permute(0, 3, 1, 2)
        z = self.net(z).squeeze(2)
        return self.head(z)


class GraphAttention(nn.Module):
    def __init__(self, adj):
        super().__init__()
        self.register_buffer("A", adj > 0)
        self.q = nn.Linear(2, 8)
        self.k = nn.Linear(2, 8)
        self.v = nn.Linear(2, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        z = x[:, -1]
        q = self.q(z)
        k = self.k(z)
        v = self.v(z)

        scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(8)
        scores = scores.masked_fill(~self.A.unsqueeze(0), -1e4)
        attn = scores.softmax(-1)

        h = torch.matmul(attn, v)
        return self.head(h).permute(0, 2, 1)


class Fusion(nn.Module):
    def __init__(self, graph, lstm):
        super().__init__()
        self.graph = graph
        self.lstm = lstm

    def forward(self, x):
        return 0.5 * self.graph(x) + 0.5 * self.lstm(x)


def masked_loss(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1)


def fit_smoke(
    name, model,
    xtr, ytr, mtr,
    xva, yva, mva,
    xte
):
    start = time.perf_counter()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Only two updates: smoke testing, not full training.
    for _ in range(2):
        optimizer.zero_grad()
        loss = masked_loss(model(xtr), ytr, mtr)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        val = float(masked_loss(model(xva), yva, mva))
        output = model(xte)

    if not torch.isfinite(output).all():
        raise RuntimeError(
            name + " produced non-finite test-shaped output"
        )

    print(
        f"{name}: validation_smoke_loss={val:.5f}; "
        "unscored_test_inference=finite",
        flush=True
    )

    log(name, len(xtr) + len(xva) + len(xte), start)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/metr-la.h5"
    )

    parser.add_argument(
        "--split-dir",
        default="split_ids"
    )

    parser.add_argument(
        "--graph",
        default="data/sensor_graph/adj_mx.pkl"
    )

    args = parser.parse_args()

    total_start = time.perf_counter()

    torch.set_num_threads(2)
    torch.manual_seed(22)
    np.random.seed(22)

    # Load METR-LA data.
    with h5py.File(args.data, "r") as f:
        raw = f["df"]["block0_values"][:].astype(np.float32)

    n, nodes = raw.shape

    # Read the reviewed issue-time split IDs.
    def read_ids(filename):
        path = Path(args.split_dir) / filename

        data = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8"
        )

        return np.atleast_1d(data)["issue_row"].astype(int)

    train_ids = read_ids("train_ids.csv")[:8]
    val_ids = read_ids("validation_ids.csv")[:8]
    test_ids = read_ids("test_ids.csv")[:8]

    # Training-only normalization.
    train_end = int(0.7 * n)
    train_values = raw[:train_end]

    train_values = np.where(
        train_values != 0,
        train_values,
        np.nan
    )

    mu = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)

    mu = np.nan_to_num(mu)
    std = np.nan_to_num(std)
    std = np.maximum(std, 1e-5)

    observed = raw != 0

    scaled = np.where(
        observed,
        (raw - mu) / std,
        0
    ).astype(np.float32)

    inputs = np.stack(
        [scaled, observed.astype(np.float32)],
        axis=-1
    )

    # Construct input histories and training/validation targets.
    def make_samples(issue_times):
        x = np.stack([
            inputs[t - 11:t + 1]
            for t in issue_times
        ])

        y = np.stack([
            scaled[[t + 3, t + 6, t + 12]]
            for t in issue_times
        ])

        mask = np.stack([
            observed[[t + 3, t + 6, t + 12]]
            for t in issue_times
        ]).astype(np.float32)

        return (
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(mask)
        )

    x_train, y_train, m_train = make_samples(train_ids)
    x_val, y_val, m_val = make_samples(val_ids)

    # Test-shaped input only. No test targets are loaded.
    x_test = torch.tensor(
        np.stack([
            inputs[t - 11:t + 1]
            for t in test_ids
        ])
    )

    print(
        f"data/split loading: train={len(train_ids)} "
        f"validation={len(val_ids)} "
        f"test_inputs={len(test_ids)} nodes={nodes}",
        flush=True
    )

    log(
        "data/split loading",
        len(train_ids) + len(val_ids) + len(test_ids),
        total_start
    )

    # Non-DL smoke baseline: training-only per-sensor mean.
    start = time.perf_counter()

    baseline = torch.tensor(
        np.broadcast_to(
            mu,
            (len(y_val), 3, nodes)
        ).copy()
    )

    print(
        "non-DL baseline: ran; "
        f"validation_tensor_shape={tuple(baseline.shape)}",
        flush=True
    )

    log("non-DL baseline", len(y_val), start)

    # Load graph adjacency matrix.
    with open(args.graph, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    adj = torch.tensor(
        np.asarray(obj[2], dtype=np.float32)
    )

    adj = adj / adj.sum(
        1,
        keepdims=True
    ).clip(min=1e-8)

    # 1-D GCN + GRU.
    graph = GraphGRU(nodes, adj)

    fit_smoke(
        "1-D GCN+GRU",
        graph,
        x_train, y_train, m_train,
        x_val, y_val, m_val,
        x_test
    )

    # Shared per-sensor LSTM.
    lstm = SharedLSTM()

    fit_smoke(
        "1-D shared LSTM",
        lstm,
        x_train, y_train, m_train,
        x_val, y_val, m_val,
        x_test
    )

    # 2-D sensor-time CNN.
    cnn = SensorTimeCNN()

    fit_smoke(
        "2-D sensor-time CNN",
        cnn,
        x_train, y_train, m_train,
        x_val, y_val, m_val,
        x_test
    )

    # Graph attention.
    gat = GraphAttention(adj)

    fit_smoke(
        "graph attention",
        gat,
        x_train, y_train, m_train,
        x_val, y_val, m_val,
        x_test
    )

    # Attention ablation: remove attention.
    class AttentionAblation(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 8)
            self.head = nn.Linear(8, 3)

        def forward(self, x):
            return self.head(
                self.proj(x[:, -1])
            ).permute(0, 2, 1)

    ablation = AttentionAblation()

    fit_smoke(
        "attention ablation (no attention)",
        ablation,
        x_train, y_train, m_train,
        x_val, y_val, m_val,
        x_test
    )

    # Fusion: GCN+GRU and shared LSTM.
    fusion = Fusion(graph, lstm)

    start = time.perf_counter()
    fusion.eval()

    with torch.no_grad():
        val_output = fusion(x_val)
        test_output = fusion(x_test)

    if not torch.isfinite(test_output).all():
        raise RuntimeError("Fusion produced non-finite output")

    print(
        "fusion GCN+GRU + shared LSTM: "
        f"validation_shape={tuple(val_output.shape)}; "
        "unscored_test_inference=finite",
        flush=True
    )

    log(
        "fusion",
        len(x_val) + len(x_test),
        start
    )

    print(
        f"TOTAL ELAPSED: "
        f"{time.perf_counter() - total_start:.2f}s",
        flush=True
    )

    print("TEST METRICS: NOT COMPUTED", flush=True)


if __name__ == "__main__":
    main()
