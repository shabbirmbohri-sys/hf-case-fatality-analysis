"""
Standard errors / p-values for the spatial error model (conditional on lambda_hat --
a standard approximation that does not propagate lambda's own estimation uncertainty;
flagged as a simplification), plus VIFs and Getis-Ord Gi* on the fully adjusted model.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import math
import numpy as np
from spatial_lib import spatial_cochrane_orcutt, vif, getis_ord_gi_star, norm_sf_two_sided

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W = np.load(f"{BASE}/data/W_matrix.npy")
d = np.load(f"{BASE}/results/primary_fit.npz")

X_full, y, w = d["X_full"], d["y"], d["w"]
n = X_full.shape[0]
lam = float(d["sem_lambda_full"])
beta = d["sem_beta_full"]

covariates = [
    "unemployment", "no_hs_diploma", "pct_black", "hospitals", "pop_per_pcp",
    "poverty", "uninsured_under65", "pct_hispanic", "urban_rural",
    "diuretic_nonadherence", "ras_nonadherence", "pm25", "heart_disease_prev",
    "cardiac_rehab_participation", "ma_penetration",
]

sw = np.sqrt(w)
Xs, ys, A = spatial_cochrane_orcutt(X_full * sw[:, None], y * sw, W, lam)
resid = ys - Xs @ beta
k = X_full.shape[1]
sigma2 = (resid @ resid) / n
XtX_inv = np.linalg.inv(Xs.T @ Xs)
cov_beta = sigma2 * XtX_inv
se = np.sqrt(np.diag(cov_beta))
z = beta / se
pvals = np.array([norm_sf_two_sided(zz) for zz in z])

print("=== Spatial error model coefficients (full: 15 covariates + state FE), conditional-on-lambda SE ===")
print(f"{'Predictor':30s} {'coef(logit)':>12s} {'SE':>8s} {'z':>7s} {'P':>8s}  %change/SD  95% CI")
names = ["intercept"] + covariates + [f"state_{i}" for i in range(k - 1 - len(covariates))]
pct_rows = []
for i, name in enumerate(["intercept"] + covariates):
    coef = beta[i]
    s = se[i]
    pct_change = (math.exp(coef) - 1) * 100
    lo = (math.exp(coef - 1.96 * s) - 1) * 100
    hi = (math.exp(coef + 1.96 * s) - 1) * 100
    print(f"{name:30s} {coef:12.4f} {s:8.4f} {z[i]:7.2f} {pvals[i]:8.4f}  {pct_change:8.2f}  [{lo:.2f}, {hi:.2f}]")
    if name != "intercept":
        pct_rows.append((name, pct_change, lo, hi, pvals[i]))

# --- VIFs on the 15 standardized covariates (predictors only, no state dummies -- VIF
# reported in the source study was on the covariate block) ---
Xcov = X_full[:, 1:1+len(covariates)]
vifs = vif(Xcov)
print("\n=== Variance Inflation Factors (15 covariates) ===")
for name, v in zip(covariates, vifs):
    flag = "  <-- >5" if v > 5 else ""
    print(f"{name:30s} VIF={v:.2f}{flag}")

# --- Getis-Ord Gi* on fully adjusted (spatial) model residuals ---
W_bin = (W > 0).astype(float)
np.fill_diagonal(W_bin, 1.0)  # Gi* includes self
resid_for_gi = d["sem_beta_full"]  # placeholder, replaced below
resid_for_gi = ys - Xs @ beta  # filtered residual space is not directly comparable; use raw resid
resid_raw = y - X_full @ beta
Gi = getis_ord_gi_star(resid_raw, W_bin)
sig_high = int((Gi > 1.96).sum())
sig_low = int((Gi < -1.96).sum())
print(f"\n=== Getis-Ord Gi* on fully adjusted residuals ===")
print(f"Significant high clusters (Gi*>1.96): {sig_high}")
print(f"Significant low clusters (Gi*<-1.96): {sig_low}")

np.savez(f"{BASE}/results/inference_results.npz",
         beta=beta, se=se, z=z, pvals=pvals, vifs=vifs, Gi=Gi,
         covariates=np.array(covariates))
print("\nSaved inference_results.npz")
