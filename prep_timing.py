import time, h5py, numpy as np
t0 = time.perf_counter()
X = h5py.File("data/metr-la.h5")["df"]["block0_values"][:].astype(np.float32)
N, S = X.shape
b1, b2 = int(0.7 * N), int(0.8 * N)
obs = X != 0
cnt = np.maximum(obs[:b1].sum(0), 1)
mean = (X[:b1] * obs[:b1]).sum(0) / cnt
std = np.sqrt((((X[:b1] - mean) ** 2) * obs[:b1]).sum(0) / cnt)
std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
Z = np.where(obs, (X - mean) / std, 0.0).astype(np.float32)
INP = np.stack([Z, obs.astype(np.float32)], -1)
H = 12
def its(lo, hi, st, first=0):
    return [t for t in range(max(first, lo - 3), hi - 12) if t - H + 1 >= 0][::st]
for name, ts in [("train", its(0, b1, 6, 11)), ("val", its(b1, b2, 12)), ("test", its(b2, N, 12))]:
    np.stack([INP[t - H + 1:t + 1] for t in ts])
print("preprocessing elapsed: %.2f s" % (time.perf_counter() - t0))