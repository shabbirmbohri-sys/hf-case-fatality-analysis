"""
Sensitivity analysis 2: exclude counties nearest the CDC suppression threshold
(smallest 15% by estimated hospitalization volume).
Sensitivity analysis 3: leave-one-state-out refitting (non-spatial WLS, for speed --
spatial refit 50x would require 50 SEM optimizations; report both where feasible).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import math
import numpy as np
from spatial_lib import wls_fit, fit_sem, spatial_cochrane_orcutt, norm_sf_two_sided

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
W_full_matrix = np.load(f"{BASE}/data/W_matrix.npy")

with open(f"{BASE}/data/analytic_sample_v2_final.csv") as f:
    r = csv.DictReader(f)
    rows = list(r)
n = len(rows)

covariates = [
    "unemployment", "no_hs_diploma", "pct_black", "hospitals", "pop_per_pcp",
    "poverty", "uninsured_under65", "pct_hispanic", "urban_rural",
    "diuretic_nonadherence", "ras_nonadherence", "pm25", "heart_disease_prev",
    "cardiac_rehab_participation", "ma_penetration",
]
focus = ["poverty", "urban_rural", "pct_black", "pm25", "no_hs_diploma", "heart_disease_prev"]

y_pct_all = np.array([float(r["hf_case_fatality"]) for r in rows]) / 100.0
y_all = np.log(y_pct_all / (1 - y_pct_all))
Xcov_raw_all = np.array([[float(r[c]) for c in covariates] for r in rows])
vol_all = np.array([float(r["est_hosp_volume"]) for r in rows])
states_all = np.array([r["State"] for r in rows])

# ============ Sensitivity 2: exclude smallest 15% by volume ============
threshold = np.percentile(vol_all, 15)
keep_mask = vol_all > threshold
n_kept = keep_mask.sum()
print(f"=== Suppression-threshold sensitivity: excluding smallest 15% by volume ===")
print(f"Threshold (15th pct of est_hosp_volume): {threshold:.1f}")
print(f"Counties retained: {n_kept} of {n} (excluded {n - n_kept})")

def build_design(idx, ref_state=None):
    sub_rows = [rows[i] for i in idx]
    y_pct = np.array([float(r["hf_case_fatality"]) for r in sub_rows]) / 100.0
    y = np.log(y_pct / (1 - y_pct))
    Xraw = np.array([[float(r[c]) for c in covariates] for r in sub_rows])
    Xstd = (Xraw - Xraw.mean(axis=0)) / Xraw.std(axis=0, ddof=0)
    states = sorted(set(r["State"] for r in sub_rows))
    ref = ref_state if (ref_state in states) else states[0]
    dummies = [s for s in states if s != ref]
    Xstate = np.zeros((len(sub_rows), len(dummies)))
    for i, r in enumerate(sub_rows):
        if r["State"] != ref:
            Xstate[i, dummies.index(r["State"])] = 1.0
    vol = np.array([float(r["est_hosp_volume"]) for r in sub_rows])
    intercept = np.ones((len(sub_rows), 1))
    X = np.hstack([intercept, Xstd, Xstate])
    return X, y, vol, len(sub_rows)

idx_kept = np.where(keep_mask)[0]
X_sub, y_sub, vol_sub, n_sub = build_design(idx_kept)
w_sub = vol_sub / vol_sub.mean()
beta_sub, _ = wls_fit(X_sub, y_sub, w_sub)

print(f"\nNon-spatial WLS (state FE + 15 covariates), n={n_sub}:")
for c in focus:
    i = 1 + covariates.index(c)
    pct = (math.exp(beta_sub[i]) - 1) * 100
    print(f"  {c:22s}: {pct:+.2f}% per SD")

print("\n(compare to primary-sample non-spatial coefficients below)")
Xstd_all = (Xcov_raw_all - Xcov_raw_all.mean(axis=0)) / Xcov_raw_all.std(axis=0, ddof=0)
states_u = sorted(set(states_all))
ref = states_u[0]
dummies = [s for s in states_u if s != ref]
Xstate_all = np.zeros((n, len(dummies)))
for i, s in enumerate(states_all):
    if s != ref:
        Xstate_all[i, dummies.index(s)] = 1.0
X_all_full = np.hstack([np.ones((n, 1)), Xstd_all, Xstate_all])
w_all = vol_all / vol_all.mean()
beta_all, _ = wls_fit(X_all_full, y_all, w_all)
for c in focus:
    i = 1 + covariates.index(c)
    pct = (math.exp(beta_all[i]) - 1) * 100
    print(f"  {c:22s}: {pct:+.2f}% per SD")

# ============ Sensitivity 3: leave-one-state-out (non-spatial WLS) ============
print("\n\n=== Leave-one-state-out (non-spatial WLS, state FE + 15 covariates) ===")
states_unique = sorted(set(states_all))
loso_results = {c: [] for c in focus}
for st in states_unique:
    idx = np.where(states_all != st)[0]
    if len(idx) < 100:
        continue
    X_loso, y_loso, vol_loso, n_loso = build_design(idx, ref_state=None)
    w_loso = vol_loso / vol_loso.mean()
    try:
        beta_loso, _ = wls_fit(X_loso, y_loso, w_loso)
    except np.linalg.LinAlgError:
        continue
    for c in focus:
        i = 1 + covariates.index(c)
        pct = (math.exp(beta_loso[i]) - 1) * 100
        loso_results[c].append(pct)

print(f"Refit {len(loso_results[focus[0]])} times (once per state with >=100 remaining counties)")
print(f"{'Predictor':22s} {'full-sample %':>14s} {'LOSO min':>10s} {'LOSO max':>10s} {'LOSO range':>12s}")
for c in focus:
    i = 1 + covariates.index(c)
    full_pct = (math.exp(beta_all[i]) - 1) * 100
    vals = loso_results[c]
    print(f"{c:22s} {full_pct:14.2f} {min(vals):10.2f} {max(vals):10.2f} {max(vals)-min(vals):12.2f}")

np.savez(f"{BASE}/results/sensitivity_suppression_loso.npz",
         beta_suppression_excl=beta_sub, n_after_exclusion=n_sub,
         **{f"loso_{c}": np.array(loso_results[c]) for c in focus})
print("\nSaved sensitivity_suppression_loso.npz")
