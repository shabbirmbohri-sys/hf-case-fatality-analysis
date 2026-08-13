"""
Core estimators, implemented directly on numpy (no scipy/statsmodels available).
- Weighted OLS
- Moran's I (permutation-based, two-sided)
- Spatial error model via concentrated ML (log-determinant via np.linalg.slogdet,
  NOT via eigenvalue symmetrization -- see protocol Section 7 warning)
- Getis-Ord Gi*
- Normal tail probabilities via math.erf (no scipy.stats needed)
"""
import numpy as np
import math


def norm_sf_two_sided(z):
    """Two-sided p-value from a standard normal z-score, using math.erf."""
    z = abs(z)
    # P(|Z| > z) = 2 * (1 - Phi(z)) = erfc(z/sqrt(2))
    return math.erfc(z / math.sqrt(2))


def wls_fit(X, y, w):
    """Weighted least squares via normal equations. w = precision weights (n,)."""
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta
    return beta, resid


def ols_fit(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def loglik_wls(X, y, w, beta, resid, n, k):
    """Gaussian log-likelihood for weighted regression (weights = precision, i.e.
    variance_i = sigma2 / w_i)."""
    wresid2 = w * resid**2
    sigma2 = wresid2.sum() / n
    ll = (
        -n / 2 * math.log(2 * math.pi)
        - n / 2 * math.log(sigma2)
        + 0.5 * np.log(w).sum()
        - n / 2
    )
    return ll, sigma2


def aic_bic(loglik, k, n):
    aic = 2 * k - 2 * loglik
    bic = k * math.log(n) - 2 * loglik
    return aic, bic


def morans_i(resid, W, n_perm=999, rng=None, two_sided=True):
    """Global Moran's I on a residual vector, with permutation-based p-value."""
    if rng is None:
        rng = np.random.default_rng(20260812)
    n = len(resid)
    z = resid - resid.mean()
    S0 = W.sum()
    numer = z @ (W @ z)
    denom = z @ z
    I_obs = (n / S0) * (numer / denom)

    I_perm = np.empty(n_perm)
    for p in range(n_perm):
        zp = rng.permutation(z)
        I_perm[p] = (n / S0) * ((zp @ (W @ zp)) / (zp @ zp))

    if two_sided:
        # two-sided permutation p-value: proportion at least as extreme in |.|
        more_extreme = np.sum(np.abs(I_perm) >= abs(I_obs))
        p_value = (more_extreme + 1) / (n_perm + 1)
    else:
        more_extreme = np.sum(I_perm >= I_obs)
        p_value = (more_extreme + 1) / (n_perm + 1)

    return I_obs, p_value, I_perm


def spatial_cochrane_orcutt(X, y, W, lam):
    """Apply (I - lam*W) transform to X and y."""
    n = X.shape[0]
    A = np.eye(n) - lam * W
    return A @ X, A @ y, A


def sem_concentrated_negloglik(lam, X, y, W, n):
    """Negative concentrated log-likelihood for the spatial error model at a given lambda."""
    Xs, ys, A = spatial_cochrane_orcutt(X, y, W, lam)
    beta, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
    resid = ys - Xs @ beta
    sigma2 = (resid @ resid) / n
    if sigma2 <= 0:
        return np.inf
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        # (I - lam W) should have positive determinant sign in the feasible region;
        # if not, this lambda is out of the valid range
        return np.inf
    ll = -n / 2 * math.log(2 * math.pi) - n / 2 * math.log(sigma2) + logdet - n / 2
    return -ll


def golden_section_search(f, a, b, tol=1e-6, max_iter=200):
    """1-D minimizer, golden-section search (no scipy available)."""
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    xmin = (a + b) / 2
    return xmin, f(xmin)


def fit_sem(X, y, W, lam_bounds=(-0.95, 0.95)):
    n, k = X.shape
    f = lambda lam: sem_concentrated_negloglik(lam, X, y, W, n)
    lam_hat, negll = golden_section_search(f, lam_bounds[0], lam_bounds[1], tol=1e-7)
    Xs, ys, A = spatial_cochrane_orcutt(X, y, W, lam_hat)
    beta, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
    resid_filtered = ys - Xs @ beta  # this is epsilon = (I-lam W)(y - X beta_orig)? careful:
    # Note: resid_filtered here = (I-lam W)y - (I-lam W)X beta = (I-lam W)(y - X beta)
    sigma2 = (resid_filtered @ resid_filtered) / n
    loglik = -negll
    # raw-scale residual (for Moran's I on non-transformed residuals, and for filtered check)
    resid_raw = y - X @ beta
    return {
        "lambda": lam_hat,
        "beta": beta,
        "sigma2": sigma2,
        "loglik": loglik,
        "resid_raw": resid_raw,
        "resid_filtered": resid_filtered,  # (I - lam W) applied to raw residuals
    }


def getis_ord_gi_star(values, W_bin_with_self):
    """Getis-Ord Gi* using a binary contiguity matrix that includes self (1 on diagonal)."""
    n = len(values)
    xbar = values.mean()
    s = values.std(ddof=0)
    Gi = np.zeros(n)
    for i in range(n):
        wi = W_bin_with_self[i, :]
        Wi_sum = wi.sum()
        numer = (wi * values).sum() - xbar * Wi_sum
        denom = s * math.sqrt((n * (wi**2).sum() - Wi_sum**2) / (n - 1))
        Gi[i] = numer / denom if denom > 0 else 0.0
    return Gi


def vif(X_no_intercept):
    """Variance inflation factor for each column of X (excluding intercept/dummies),
    via R^2 of regressing each column on the rest."""
    n, k = X_no_intercept.shape
    vifs = np.zeros(k)
    for j in range(k):
        y = X_no_intercept[:, j]
        Xother = np.delete(X_no_intercept, j, axis=1)
        Xother_i = np.column_stack([np.ones(n), Xother])
        beta, _, _, _ = np.linalg.lstsq(Xother_i, y, rcond=None)
        pred = Xother_i @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs[j] = 1 / (1 - r2) if r2 < 1 else np.inf
    return vifs
