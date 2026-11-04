import sys, numpy as np, h5py
H5 = sys.argv[1]
X = h5py.File(H5)["df"]["block0_values"][:]
N, S = X.shape
obs = X != 0
b1, b2 = int(0.7 * N), int(0.8 * N)
print("N", N, "boundaries", b1, b2)

# --- DCRNN published split (scripts/generate_training_data.py, commit 602afd9) ---
nw = N - 11 - 12 + 0            # t runs 11 .. N-13  (max_t = N-12 exclusive)
t_all = np.arange(11, N - 12)
num = len(t_all); n_test = round(num * 0.2); n_train = round(num * 0.7); n_val = num - n_test - n_train
tr, va, te = t_all[:n_train], t_all[n_train:n_train + n_val], t_all[-n_test:]
tg = lambda ts: set((ts[:, None] + np.arange(1, 13)).ravel().tolist())     # DCRNN predicts all 12 steps
print("DCRNN windows", num, "train/val/test", n_train, n_val, n_test)
print("shared target rows train/val:", len(tg(tr) & tg(va)), "| val/test:", len(tg(va) & tg(te)))
print("last train t", tr[-1], "first val t", va[0], "last val t", va[-1], "first test t", te[0])

# --- Annex D4 issue times under two readings of "first eligible" ---
def issue(lo, hi, stride, first):
    ts = [t for t in range(first, hi - 12)]           # last target t+12 must be < hi
    return np.array(ts[::stride])
H_ = [3, 6, 12]
def support(ts, sensors):
    return [int(obs[ts + h][:, sensors].sum()) for h in H_]
allS, s20 = np.arange(S), np.arange(20)
for name, start in (("start at boundary row (accepted)", b1), ("start at boundary-3 (history crosses)", b1 - 3)):
    ts = issue(b1, b2, 12, start)
    print(f"VAL {name}: n={len(ts)} first={ts[0]} | 207 sensors {support(ts, allS)} | first 20 {support(ts, s20)}")
for name, start in (("start at boundary row (accepted)", b2), ("start at boundary-3", b2 - 3)):
    ts = issue(b2, N, 12, start)
    print(f"TEST {name}: n={len(ts)} first={ts[0]}")
for name, start in (("train start 11 (annex)", 11), ("train start 12", 12)):
    ts = issue(0, b1, 6, start)
    print(f"TRAIN {name}: n={len(ts)} first={ts[0]} last={ts[-1]} last target={ts[-1]+12}")
