# hf-case-fatality-analysis
# Heart Failure In-Hospital Case Fatality: Geography Above the County

Analysis code and data supporting the manuscript "Geography Above the County: State and
Regional Structure in Heart Failure In-Hospital Case Fatality Among Medicare Fee-for-Service
Beneficiaries Aged 65 Years or Older, 486 US Counties, 2019-2021," prepared for submission to
*Preventing Chronic Disease*.

This repository is deposited so the spatial error model implementation described in the
manuscript's Methods can be checked independently, per the manuscript's stated Limitation that
established spatial econometrics packages (R's `spatialreg`/`spdep`, Python's `PySAL`) were not
available in the computing environment used for this analysis.

## Scope

This repository covers everything from the final analytic dataset (`data/analytic_sample_v2_final.csv`)
through every statistical result and figure reported in the manuscript. It does **not** include
the earlier data-acquisition step (querying the CDC Interactive Atlas of Heart Disease and Stroke
API and merging in CMS Medicare Advantage enrollment files), which was done interactively rather
than as a standalone script. The analytic dataset itself is included so that step does not need
to be reproduced to check the statistical results.

## Structure

```
data/
  analytic_sample_v2_final.csv   486 counties x 22 columns: outcome, 15 covariates,
                                  MA penetration inputs, analytic weight inputs
  W_matrix.npy                   486x486 row-standardized queen contiguity spatial
                                  weights matrix (precomputed)
  fips_order.csv                 row order of W_matrix.npy / analytic_sample_v2_final.csv
  raw/county_adjacency2023.txt   source Census Bureau county adjacency file (public domain),
                                  used by build_adjacency.py to regenerate W_matrix.npy
  counties-albers-10m.json       US county/state boundaries (TopoJSON) used only for
                                  Figure 1; from us-atlas (MIT license),
                                  https://github.com/topojson/us-atlas

analysis/
  spatial_lib.py                     Core estimators (weighted OLS, Moran's I, spatial
                                      error model via concentrated ML, Getis-Ord Gi*, VIF).
                                      Implemented directly on numpy -- no scipy/statsmodels/
                                      PySAL were available in the original computing
                                      environment. Log-determinant uses np.linalg.slogdet
                                      directly, NOT a symmetrized-eigenvalue shortcut, which
                                      is invalid for a row-standardized (non-symmetric) W.
  validate_simulation.py             Mandatory pre-registration-style validation: recovers
                                      known lambda/beta from simulated data on the real
                                      486-county W, and checks the slogdet vs. wrong-shortcut
                                      divergence directly.
  build_adjacency.py                 Builds W_matrix.npy and fips_order.csv from the raw
                                      Census adjacency file, restricted to the analytic sample.
  fit_primary_model.py               Fits the non-spatial (county-only/state-only/full) and
                                      spatial error models; saves results/primary_fit.npz.
  inference_and_vif.py               Standard errors (conditional on lambda-hat), VIFs, and
                                      Getis-Ord Gi* on the fully adjusted model residuals.
  sensitivity_weighting.py           Refits under 4 weighting schemes (volume/primary,
                                      unweighted, sqrt-volume, log-volume).
  sensitivity_suppression_and_loso.py  Excludes counties nearest the CDC suppression
                                      threshold; refits leaving out one state at a time (50x).
  render_getis_ord_map.py            Renders Figure 1. No geopandas/shapely were available in
                                      the original environment, so the TopoJSON county/state
                                      boundaries are decoded by hand (arc delta-decoding +
                                      stitching) and drawn with matplotlib PathPatch/
                                      PatchCollection.

results/
  primary_fit.npz, inference_results.npz, sensitivity_weighting.npz,
  sensitivity_suppression_loso.npz   Saved numeric outputs, as produced by the scripts above.
                                      Included so every number in the manuscript can be
                                      checked without re-running the pipeline.

figures/
  Figure1_GetisOrd_map.png           As it appears in the manuscript.
```

## Running the pipeline

Requires Python 3 with `numpy` and (for the figure only) `matplotlib`. No other third-party
packages are used anywhere in this repository.

```
python3 analysis/build_adjacency.py              # optional -- W_matrix.npy is already provided
python3 analysis/validate_simulation.py          # estimator validation on simulated data
python3 analysis/fit_primary_model.py            # primary non-spatial + spatial error models
python3 analysis/inference_and_vif.py            # SEs, VIFs, Getis-Ord Gi*
python3 analysis/sensitivity_weighting.py        # 4-scheme weighting sensitivity
python3 analysis/sensitivity_suppression_and_loso.py
python3 analysis/render_getis_ord_map.py         # Figure 1
```

Each script writes its outputs to `results/` (or `figures/` for the map) and can be re-run
independently once `data/W_matrix.npy` and `results/primary_fit.npz` exist. All scripts locate
the repository root relative to their own file location, so this repository can be run from any
path without editing.

## A note on the spatial error model implementation

The estimator in `spatial_lib.py` fits `y = X*beta + u`, `u = lambda*W*u + epsilon` by maximum
likelihood, concentrating the log-likelihood over the single spatial parameter `lambda` and
solving it with a golden-section search over `(-0.95, 0.95)` (no `scipy.optimize` available).
`validate_simulation.py` was run before this estimator was ever applied to the real data: on
simulated data using the actual 486-county adjacency structure, with a known `lambda = 0.55` and
known regression coefficients, the estimator recovered `lambda = 0.515` and coefficients within
about 0.13 of their true values. It also confirms directly that `np.linalg.slogdet` and a
"symmetrize-then-eigendecompose" shortcut diverge substantially at this `W` (by up to ~27
log-units), which is expected because row-standardized contiguity matrices are not symmetric --
the shortcut is only valid for a symmetric `W`, and `spatial_lib.py` does not use it.

## AI-use disclosure

Generative AI (Claude, Anthropic) was used to implement, validate, and refactor the code in this
repository, and to write this README, based on an analysis pipeline and results developed and
reviewed by the human author(s) of the associated manuscript. See the manuscript's AI Use
Disclosure section for the full statement.
