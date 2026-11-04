"""Item 7a: one timed fit of the reference configuration for card D1 -- GCN+GRU on METR-LA.
Run from the DCRNN repo root (CPU PyTorch is enough):
    pip install torch h5py numpy psutil
    python gcngru_timed_run.py                 # default: seed 22, <=30 epochs or <=30 min, patience 5
    python gcngru_timed_run.py --threads 6     # optional: pin torch threads, and report that setting
Test LABELS are never loaded: test-shaped INPUTS are only run forward once, to time inference.
"""
import argparse, time, os, sys, json, pickle, platform, random
import numpy as np, h5py, psutil, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--threads", type=int, default=0, help="0 = leave PyTorch default")
p.add_argument("--max_epochs", type=int, default=30)
p.add_argument("--max_minutes", type=float, default=30.0)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=22)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
if a.threads > 0:
    torch.set_num_threads(a.threads)
proc = psutil.Process()

def peak_mem_mb():
    mi = proc.memory_info()
    pk = getattr(mi, "peak_wset", None)                       # Windows
    if pk is None:
        try:
            import resource                                      # Linux/macOS
            pk = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except Exception:
            pk = mi.rss
    return pk / 2**20

# ---- DATA PREP (Annex D4) ----
H, T_OUT = 12, (3, 6, 12)
Xraw = h5py.File(f"{a.root}/data/metr-la.h5")["df"]["block0_values"][:].astype(np.float32)
N, S = Xraw.shape
b1, b2 = int(0.7 * N), int(0.8 * N)
obs = Xraw != 0                                                  # zero = missing, not speed 0
cnt = np.maximum(obs[:b1].sum(0), 1)
mean = (Xraw[:b1] * obs[:b1]).sum(0) / cnt                       # scaler fitted on TRAIN only
std = np.sqrt((((Xraw[:b1] - mean) ** 2) * obs[:b1]).sum(0) / cnt)
std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
Z = np.where(obs, (Xraw - mean) / std, 0.0).astype(np.float32)   # missing -> 0 after scaling
INP = np.stack([Z, obs.astype(np.float32)], -1)                  # (N, S, 2): value + observed indicator

# Load the instructor-reviewed issue times directly from split CSVs.
def load_issue_times(filename):
    path = os.path.join(a.root, "split_ids", filename)
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(rows)["issue_row"].astype(int).tolist()

ts_tr = load_issue_times("train_ids.csv")
ts_va = load_issue_times("validation_ids.csv")
ts_te = load_issue_times("test_ids.csv")

assert len(ts_tr) == 3995
assert len(ts_va) == 285
assert len(ts_te) == 571

print(
    f"Loaded official issue times: "
    f"train={len(ts_tr)}, validation={len(ts_va)}, test={len(ts_te)}"
)

def make_x(ts):
    return torch.from_numpy(np.stack([INP[t - H + 1:t + 1] for t in ts]))          # (B, 12, S, 2)

def make_y(ts):
    y = np.stack([Z[[t + k for k in T_OUT]] for t in ts])                           # (B, 3, S)
    m = np.stack([obs[[t + k for k in T_OUT]] for t in ts]).astype(np.float32)
    return torch.from_numpy(y), torch.from_numpy(m)
# ---- END DATA PREP ----

Xtr, (Ytr, Mtr) = make_x(ts_tr), make_y(ts_tr)
Xva, (Yva, Mva) = make_x(ts_va), make_y(ts_va)
Xte = make_x(ts_te)                                              # inputs only -- no test targets are built

ids, _, A = pickle.load(open(f"{a.root}/data/sensor_graph/adj_mx.pkl", "rb"), encoding="latin1")
A = np.asarray(A, dtype=np.float32)
A = A / np.maximum(A.sum(1, keepdims=True), 1e-8)               # row-normalised, directed, self-loops as shipped

class GCNGRU(nn.Module):
    """Graph conv on every time step (weights shared across nodes) -> GRU per node (shared) -> 3 horizons."""
    def __init__(s, A, fin=2, gc=32, hid=64, nout=3):
        super().__init__()
        s.register_buffer("A", torch.from_numpy(A))
        s.gc, s.gru, s.head = nn.Linear(fin, gc), nn.GRU(gc, hid, batch_first=True), nn.Linear(hid, nout)
    def forward(s, x):                                            # x: (B, T, S, F)
        B, T, S_, F = x.shape
        h = torch.relu(s.gc(torch.einsum("ij,btjf->btif", s.A, x)))
        h = h.permute(0, 2, 1, 3).reshape(B * S_, T, -1)
        o, _ = s.gru(h)
        return s.head(o[:, -1]).reshape(B, S_, -1).permute(0, 2, 1)   # (B, 3, S)

model = GCNGRU(A)
n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)

def masked_mae(pred, y, m):
    return ((pred - y).abs() * m).sum() / m.sum().clamp(min=1.0)

@torch.no_grad()
def evaluate(X, Y, M):
    model.eval(); tot = cnt_ = 0.0
    for i in range(0, len(X), 64):
        pr = model(X[i:i + 64]); m = M[i:i + 64]
        tot += ((pr - Y[i:i + 64]).abs() * m).sum().item(); cnt_ += m.sum().item()
    return tot / max(cnt_, 1.0)

print("python", sys.version.split()[0], "|", platform.platform(), "| torch", torch.__version__, "| numpy", np.__version__, "| h5py", h5py.__version__)
print("cores logical/physical", psutil.cpu_count(), psutil.cpu_count(logical=False), "| RAM GB %.1f" % (psutil.virtual_memory().total / 2**30))
print("torch threads", torch.get_num_threads(), "| interop", torch.get_num_interop_threads(), "| data-loader workers 0 (in-memory tensors)")
print("issue times train/val/test", len(ts_tr), len(ts_va), len(ts_te), "| trainable params", n_params, "(limit 250,000)")
assert n_params <= 250_000

ckpt = os.path.join(a.root, "gcngru_best.pt")
log, best, bad, updates, stop_reason = [], float("inf"), 0, 0, None
g = torch.Generator().manual_seed(a.seed)
w0, c0 = time.perf_counter(), time.process_time()               # clock starts at the first training batch
limit_s = a.max_minutes * 60
for ep in range(1, a.max_epochs + 1):
    e0 = time.perf_counter(); model.train(); perm = torch.randperm(len(Xtr), generator=g); run = nb = 0
    for i in range(0, len(Xtr), a.batch):
        b = perm[i:i + a.batch]
        loss = masked_mae(model(Xtr[b]), Ytr[b], Mtr[b])
        opt.zero_grad(); loss.backward(); opt.step()
        run += loss.item(); nb += 1; updates += 1
        if time.perf_counter() - w0 >= limit_s:
            stop_reason = "minutes"; break
    vloss = evaluate(Xva, Yva, Mva)                                # validation pass is inside the clock
    if vloss < best:
        best, bad = vloss, 0; torch.save(model.state_dict(), ckpt)  # checkpoint save is inside the clock
    else:
        bad += 1
    log.append((ep, run / max(nb, 1), vloss, time.perf_counter() - e0))
    print("epoch %2d | train MAE(scaled) %.4f | val MAE(scaled) %.4f | %.1fs | cumulative %.1fs | updates %d"
          % (ep, run / max(nb, 1), vloss, time.perf_counter() - e0, time.perf_counter() - w0, updates))
    if stop_reason: break
    if bad >= a.patience: stop_reason = "early stopping"; break
    if time.perf_counter() - w0 >= limit_s: stop_reason = "minutes"; break
else:
    stop_reason = "epochs"
wall, cpu = time.perf_counter() - w0, time.process_time() - c0
model.load_state_dict(torch.load(ckpt)); model.eval()

i0, ic0 = time.perf_counter(), time.process_time()                # one timed inference pass, test-shaped inputs, NOT scored
with torch.no_grad():
    outs = torch.cat([model(Xte[i:i + 64]) for i in range(0, len(Xte), 64)])
inf_wall, inf_cpu = time.perf_counter() - i0, time.process_time() - ic0
finite = bool(torch.isfinite(outs).all())

summary = dict(epochs_completed=len(log), updates=updates, stopped_by=stop_reason, elapsed_s=round(wall, 1),
               cpu_s=round(cpu, 1), per_epoch_s=round(wall / len(log), 1), first_val=log[0][2], best_val=best,
               first_train=log[0][1], last_train=log[-1][1], inference_s=round(inf_wall, 2), inference_cpu_s=round(inf_cpu, 2),
               inference_examples=len(Xte), outputs_finite=finite, peak_mem_mb=round(peak_mem_mb()),
               threads=torch.get_num_threads(), params=n_params, seed=a.seed)
print("\n=== PASTE INTO ITEM 7 ===")
print(json.dumps(summary, indent=2))
with open(os.path.join(a.root, "gcngru_epoch_log.csv"), "w") as f:
    f.write("epoch,train_mae_scaled,val_mae_scaled,epoch_seconds\n")
    f.writelines("%d,%.5f,%.5f,%.2f\n" % r for r in log)
print("per-epoch losses saved to gcngru_epoch_log.csv (use it for the loss table/plot)")
