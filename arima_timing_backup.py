"""Item 7b: time the fixed ARIMA(2,0,1) bank on 20 sensors and project to 207 (Annex D4 / D0).
Run from the DCRNN repo root:   pip install statsmodels psutil h5py numpy
    python arima_timing.py
Uses TRAINING rows only for fitting; the filter-pass timing uses VALIDATION rows only (no test data is touched).
"""
import sys, time, warnings, platform
import numpy as np, h5py, psutil, statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT, K, ORDER = ".", 20, (2, 0, 1)
SENSORS = list(range(K))            # declared choice: first 20 sensors in file order

X = h5py.File(f"{ROOT}/data/metr-la.h5")["df"]["block0_values"][:].astype(np.float64)
N, S = X.shape
b1, b2 = int(0.7 * N), int(0.8 * N)
obs = X != 0
cnt = np.maximum(obs[:b1].sum(0), 1)
mean = (X[:b1] * obs[:b1]).sum(0) / cnt                        # training-observed mean per sensor
Y = np.where(obs, X, mean)                                      # impute missing with training mean

print("python", sys.version.split()[0], "|", platform.platform(), "| statsmodels", statsmodels.__version__)
print("cores (logical)", psutil.cpu_count(), "| RAM GB %.1f" % (psutil.virtual_memory().total / 2**30))

proc = psutil.Process()
fit_wall = fit_cpu = filt_wall = filt_cpu = 0.0
n_conv_warn = n_fail = 0
for j in SENSORS:
    y = Y[:b1, j]
    w0, c0 = time.perf_counter(), time.process_time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = ARIMA(y, order=ORDER, trend="c").fit()
            ok = True
        except Exception as e:
            ok = False
            print("sensor", j, "FAILED:", e)
    fit_wall += time.perf_counter() - w0
    fit_cpu += time.process_time() - c0
    if not ok:
        n_fail += 1
        continue
    if any(issubclass(c.category, ConvergenceWarning) for c in caught):
        n_conv_warn += 1
    # one state roll-forward over the validation rows only, no refit
    w0, c0 = time.perf_counter(), time.process_time()
    res.append(Y[b1:b2, j], refit=False)
    filt_wall += time.perf_counter() - w0
    filt_cpu += time.process_time() - c0

n_ok = K - n_fail
scale = S / K
print("\n--- ARIMA%s, %d sensors (fit on %d training rows each) ---" % (ORDER, K, b1))
print("fit: elapsed %.1f s, CPU %.1f s (%.2f s/sensor)" % (fit_wall, fit_cpu, fit_wall / max(n_ok, 1)))
print("fit failures:", n_fail, "| convergence warnings:", n_conv_warn)
print("projected full bank (%d sensors): elapsed %.1f s = %.1f min, CPU %.1f s" %
      (S, fit_wall * scale, fit_wall * scale / 60, fit_cpu * scale))
print("60-minute classical allowance:", "FITS" if fit_wall * scale / 60 <= 60 else "DOES NOT FIT",
      "(this is the fit alone; historical average and tuning share the same allowance)")
print("validation filter pass (%d rows): elapsed %.1f s for %d sensors -> projected %.1f s for %d sensors; "
      "test-length pass would be ~%.1fx that" % (b2 - b1, filt_wall, n_ok, filt_wall * scale, S, (N - b2) / (b2 - b1)))
print("peak memory MB:", round((getattr(proc.memory_info(), "peak_wset", None) or proc.memory_info().rss) / 2**20))
print("NOT included: 3/6/12-step forecasts at every issue time, which add to the projection.")
