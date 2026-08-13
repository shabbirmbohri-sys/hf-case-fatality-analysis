"""
Fit the primary non-spatial and spatial error models for HF in-hospital case fatality.
Outcome: logit(hf_case_fatality / 100)
Predictors: 15 standardized county covariates + state fixed effects
Weights: analytic precision weights = hf_hosp_rate * ffs_beneficiaries
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import math
import numpy as np
from spatial_lib import (
    wls_fit, ols_fit, loglik_wls, aic_bic, morans_i, fit_sem, vif, norm_sf_two_sided
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W = np.load(f"{BASE}/data/W_matrix.npy")

with open(f"{BASE}/data/analytic_sample_v2_final.csv") as f:
    r = csv.DictReader(f)
    rows = list(r)
n = len(rows)
print(f"n = {n}")

covariates = [
    "unemployment", "no_hs_diploma", "pct_black", "hospitals", "pop_per_pcp",
    "poverty", "uninsured_under65", "pct_hispanic", "urban_rural",
    "diuretic_nonadherence", "ras_nonadherence", "pm25", "heart_disease_prev",
    "cardiac_rehab_participation", "ma_penetration",
]
assert len(covariates) == 15

# outcome: logit transform
y_pct = np.array([float(r["hf_case_fatality"]) for r in rows]) / 100.0
y = np.log(y_pct / (1 - y_pct))

# covariates, standardized
Xcov_raw = np.array([[float(r[c]) for c in covariates] for r in rows])
Xcov_mean = Xcov_raw.mean(axis=0)
Xcov_sd = Xcov_raw.std(axis=0, ddof=0)
Xcov = (Xcov_raw - Xcov_mean) / Xcov_sd

# state fixed effects (drop one reference state -- use first alphabetically)
states = sorted(set(r["State"] for r in rows))
ref_state = states[0]
state_dummy_names = [s for s in states if s != ref_state]
Xstate = np.zeros((n, len(state_dummy_names)))
for i, r in enumerate(rows):
    if r["State"] != ref_state:
        j = state_dummy_names.index(r["State"])
        Xstate[i, j] = 1.0
print(f"States: {len(states)} (reference = {ref_state}), {len(state_dummy_names)} dummies")

# weights
w = np.array([float(r["est_hosp_volume"]) for r in rows])
w = w / w.mean()  # normalize for numerical stability

intercept = np.ones((n, 1))

# --- Model blocks ---
X_county_only = np.hstack([intercept, Xcov])
X_state_only = np.hstack([intercept, Xstate])
X_full = np.hstack([intercept, Xcov, Xstate])

results = {}
for name, X in [("county_only", X_county_only), ("state_only", X_state_only), ("full", X_full)]:
    beta, resid = wls_fit(X, y, w)
    k = X.shape[1]
    ll, sigma2 = loglik_wls(X, y, w, beta, resid, n, k)
    aic, bic = aic_bic(ll, k, n)
    results[name] = dict(beta=beta, resid=resid, k=k, loglik=ll, aic=aic, bic=bic, sigma2=sigma2)
    print(f"{name}: k={k}, loglik={ll:.2f}, AIC={aic:.1f}, BIC={bic:.1f}")

print("\n=== Fit comparison (Section 6: AIC/BIC, not raw LR, across differently-sized blocks) ===")
print(f"AIC improvement adding state FE to county model: "
      f"{results['county_only']['aic'] - results['full']['aic']:.1f}")
print(f"AIC improvement adding county block to state-only model: "
      f"{results['state_only']['aic'] - results['full']['aic']:.1f}")
print(f"BIC (full)={results['full']['bic']:.1f}  BIC (county_only)={results['county_only']['bic']:.1f}  "
      f"BIC (state_only)={results['state_only']['bic']:.1f}")

# --- Moran's I on non-spatial full-model residuals ---
I_obs, p_val, _ = morans_i(results["full"]["resid"], W, n_perm=999, rng=np.random.default_rng(2026))
print(f"\nMoran's I on non-spatial (full, with state FE) residuals: I={I_obs:.4f}, two-sided P={p_val:.4f}")

# --- Spatial error model: full (with state FE), PRECISION-WEIGHTED ---
# Weighting is implemented by pre-scaling X and y by sqrt(w) (mathematically identical
# to the WLS normal-equation weighting used above) before applying the concentrated SEM
# likelihood. This is an approximation: it does not perfectly preserve the spatial
# autoregressive error structure under heteroskedastic weights (sqrt(w)*u does not
# equal lam*W*(sqrt(w)*u) + sqrt(w)*eps in general unless w is constant). Flagged as a
# simplification in Methods/Limitations, consistent with how the 4-scheme sensitivity
# analysis is implemented.
sw = np.sqrt(w)
print("\n=== Fitting spatial error model (full, with state FE, precision-weighted) ===")
sem_full = fit_sem(X_full * sw[:, None], y * sw, W)
print(f"lambda = {sem_full['lambda']:.4f}")
I_filt, p_filt, _ = morans_i(sem_full["resid_filtered"], W, n_perm=999, rng=np.random.default_rng(2027))
print(f"Moran's I on spatially filtered residuals (I-lam W)u: I={I_filt:.4f}, two-sided P={p_filt:.4f}")

# --- Spatial error model: county-only (no state FE), to see lambda shrink ---
print("\n=== Fitting spatial error model (county-only, NO state FE), precision-weighted ===")
sem_nostate = fit_sem(X_county_only * sw[:, None], y * sw, W)
print(f"lambda (no state FE) = {sem_nostate['lambda']:.4f}")
print(f"lambda (with state FE) = {sem_full['lambda']:.4f}")

# save everything needed downstream
np.savez(f"{BASE}/results/primary_fit.npz",
         X_full=X_full, X_county_only=X_county_only, X_state_only=X_state_only,
         y=y, w=w, Xcov_mean=Xcov_mean, Xcov_sd=Xcov_sd,
         beta_full=results["full"]["beta"], beta_county=results["county_only"]["beta"],
         beta_state=results["state_only"]["beta"],
         sem_lambda_full=sem_full["lambda"], sem_beta_full=sem_full["beta"],
         sem_lambda_nostate=sem_nostate["lambda"],
         aic_full=results["full"]["aic"], aic_county=results["county_only"]["aic"],
         aic_state=results["state_only"]["aic"],
         bic_full=results["full"]["bic"], bic_county=results["county_only"]["bic"],
         bic_state=results["state_only"]["bic"],
         moran_I_nonspatial=I_obs, moran_p_nonspatial=p_val,
         moran_I_filtered=I_filt, moran_p_filtered=p_filt,
         )
print("\nSaved primary_fit.npz")
