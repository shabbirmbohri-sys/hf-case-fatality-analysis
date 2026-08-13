"""
Sensitivity analysis 1: four weighting schemes (protocol Section 6, mandatory).
Refits the spatial error model (state FE + 15 covariates) under:
  volume (primary), unweighted, sqrt(volume), log(volume)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import math
import numpy as np
from spatial_lib import wls_fit, fit_sem, spatial_cochrane_orcutt, norm_sf_two_sided

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W = np.load(f"{BASE}/data/W_matrix.npy")
d = np.load(f"{BASE}/results/primary_fit.npz")
X_full, y = d["X_full"], d["y"]
n = X_full.shape[0]

with open(f"{BASE}/data/analytic_sample_v2_final.csv") as f:
    r = csv.DictReader(f)
    rows = list(r)
vol = np.array([float(r["est_hosp_volume"]) for r in rows])

covariates = [
    "unemployment", "no_hs_diploma", "pct_black", "hospitals", "pop_per_pcp",
    "poverty", "uninsured_under65", "pct_hispanic", "urban_rural",
    "diuretic_nonadherence", "ras_nonadherence", "pm25", "heart_disease_prev",
    "cardiac_rehab_participation", "ma_penetration",
]

schemes = {
    "volume": vol / vol.mean(),
    "unweighted": np.ones(n),
    "sqrt_volume": np.sqrt(vol) / np.sqrt(vol).mean(),
    "log_volume": np.log(vol) / np.log(vol).mean(),
}

# focus on the headline predictors from the primary model
focus = ["no_hs_diploma", "poverty", "urban_rural", "pct_black", "pm25", "heart_disease_prev"]
focus_idx = [1 + covariates.index(c) for c in focus]  # +1 for intercept column

print("=== Sensitivity to weighting scheme (spatial error model, full spec) ===")
print(f"{'Scheme':14s}  lambda   " + "  ".join(f"{c:>16s}" for c in focus))
results = {}
for name, w in schemes.items():
    # weighted -> refit SEM using this weight vector by transforming y and X first
    # (precision-weighted spatial error: apply sqrt(w) scaling, matching primary approach
    # via WLS normal equations already embedded in wls_fit; for SEM we approximate by
    # pre-scaling X and y by sqrt(w) then running the same concentrated-likelihood SEM,
    # consistent with how the primary (volume) weighting was implemented)
    sw = np.sqrt(w)
    Xw = X_full * sw[:, None]
    yw = y * sw
    sem = fit_sem(Xw, yw, W)  # NOTE: W is unweighted adjacency; weighting enters via X,y scaling only
    lam = sem["lambda"]
    beta = sem["beta"]
    Xs, ys, A = spatial_cochrane_orcutt(Xw, yw, W, lam)
    resid = ys - Xs @ beta
    sigma2 = (resid @ resid) / n
    cov_beta = sigma2 * np.linalg.inv(Xs.T @ Xs)
    se = np.sqrt(np.diag(cov_beta))
    pct = [(math.exp(beta[i]) - 1) * 100 for i in focus_idx]
    pvals = [norm_sf_two_sided(beta[i] / se[i]) for i in focus_idx]
    results[name] = dict(lam=lam, beta=beta, se=se, pct=pct, pvals=pvals)
    row = "  ".join(f"{p:+7.2f}%(P={pv:.3f})" for p, pv in zip(pct, pvals))
    print(f"{name:14s}  {lam:.4f}   {row}")

print("\nPredictor order:", focus)
print("\n=== Interpretation guide ===")
for c in focus:
    ps = [results[s]["pvals"][focus.index(c)] for s in schemes]
    sig_count = sum(1 for p in ps if p < 0.05)
    stability = "robust (significant in all 4 schemes)" if sig_count == 4 else \
                f"weighting-sensitive (significant in {sig_count}/4 schemes)"
    print(f"{c:25s}: {stability}")

np.savez(f"{BASE}/results/sensitivity_weighting.npz",
         **{f"{name}_lam": results[name]["lam"] for name in schemes},
         **{f"{name}_beta": results[name]["beta"] for name in schemes})
print("\nSaved sensitivity_weighting.npz")
