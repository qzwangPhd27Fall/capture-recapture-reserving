"""
Doubly-robust capture-recapture estimator and improved min-chi-square.

Reference:
  Dulce Rubio & Kennedy (2024) "Population Size Estimation with Many Lists
  and Heterogeneity: A Conditional Log-Linear Model Among the Unobserved"

The doubly-robust estimator is consistent if EITHER:
  (a) the detection probability model is correctly specified, OR
  (b) the outcome regression model is correctly specified.

All CIs use nonparametric percentile bootstrap (B=500).
"""

import numpy as np
from math import comb
from typing import Optional
from scipy.optimize import minimize_scalar

from .bootstrap import (
    bootstrap_ci, parametric_bootstrap_cr, parametric_bootstrap_bc,
    jackknife_sandwich_ci, asymptotic_sandwich_ci, smoothed_frequency_bootstrap_ci,
)


def _doubly_robust_point(
    capture_histories: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    p_model: str = 'logistic',
) -> float:
    """Doubly-robust point estimate of N."""
    n_obs, D = capture_histories.shape

    if p_model == 'logistic':
        pi_hat = _estimate_pi_logistic(capture_histories, covariates)
    elif p_model == 'rf':
        pi_hat = _estimate_pi_rf(capture_histories, covariates)
    else:
        raise ValueError(f"Unknown p_model: {p_model}")

    p0_i = _estimate_p0(pi_hat, D)
    p0_avg = np.mean(p0_i)
    N_dr = n_obs / max(1.0 - p0_avg, 0.01)

    return float(N_dr)


def doubly_robust_cr(
    capture_histories: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    p_model: str = 'logistic',
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
    ci_method: str = 'asymptotic',
    verbose: bool = False,
) -> dict:
    """
    Doubly-robust population size estimator with bootstrap CI.

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    covariates : (n_obs, p) optional
    p_model : 'logistic' or 'rf'
    alpha : significance level
    n_boot : bootstrap replicates (for smoothed/parametric_bc methods)
    seed : random seed
    ci_method : 'asymptotic' (default, calibrated), 'smoothed', 'parametric_bc', 'jackknife'
    verbose : show progress bar

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, etc.
    """
    n_obs, D = capture_histories.shape

    # Estimate overdispersion for smoothed bootstrap
    f_i = capture_histories.sum(axis=1)
    p_mean = float(np.mean(f_i)) / D
    var_binom = D * p_mean * (1.0 - p_mean)
    var_obs = float(np.var(f_i, ddof=1))
    c_hat = max(1.0, var_obs / max(var_binom, 0.001))

    if ci_method == 'smoothed':
        result = smoothed_frequency_bootstrap_ci(
            _doubly_robust_point, capture_histories, covariates, p_model,
            n_boot=n_boot, alpha=alpha, seed=seed,
            lower_bound=float(n_obs), overdispersion=c_hat,
            verbose=verbose,
        )
        se_key = 'se_boot'
    elif ci_method == 'parametric_bc':
        result = parametric_bootstrap_bc(
            _doubly_robust_point, capture_histories, covariates, p_model,
            n_boot=n_boot, alpha=alpha, seed=seed,
            lower_bound=float(n_obs), verbose=verbose,
        )
        se_key = 'se_boot'
    elif ci_method == 'asymptotic':
        result = asymptotic_sandwich_ci(
            _doubly_robust_point, capture_histories, covariates, p_model,
            alpha=alpha, lower_bound=float(n_obs),
            estimator_type='doubly_robust',
            calibrate=True, n_calib=200, seed=seed, verbose=verbose,
        )
        se_key = 'se_asymptotic'
    elif ci_method == 'jackknife':
        result = jackknife_sandwich_ci(
            _doubly_robust_point, capture_histories, covariates, p_model,
            alpha=alpha, lower_bound=float(n_obs),
        )
        se_key = 'se_sandwich'
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")

    # Recompute p0_avg on original data for diagnostics
    if p_model == 'logistic':
        pi_hat = _estimate_pi_logistic(capture_histories, covariates)
    else:
        pi_hat = _estimate_pi_rf(capture_histories, covariates)
    p0_i = _estimate_p0(pi_hat, D)
    p0_avg = float(np.mean(p0_i))

    return {
        'N_hat': result['N_hat'],
        'variance': result[se_key] ** 2,
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'method': 'doubly_robust',
        'ci_method': ci_method,
        'calibration_factor': result.get('calibration_factor', 1.0),
        'pi_ever_mean': float(np.mean(1.0 - p0_i)),
        'p0_mean': p0_avg,
        'n_observed': n_obs,
        'n_ibnr': max(result['N_hat'] - n_obs, 0),
        'se_boot': result[se_key],
        'n_failed': result.get('n_failed', 0),
    }


def _min_chisq_improved_point(
    capture_histories: np.ndarray,
    K: int = 5,
) -> float:
    """
    Minimum chi-square point estimate.

    Uses the capture-frequency distribution:
      O_k = # individuals with exactly k captures (k = 1,...,D)
      Expected under model: E_k(N, p) = N · Binom(D, p, k)

    χ²(N) = Σ_{k=1}^D (O_k - N·b_k)² / (N·b_k) + penalty

    where b_k = Binom(D, p̂, k) with p̂ = f̄/D profiled out.
    The penalty ensures N predicts the observed n correctly.
    """
    n_obs, D = capture_histories.shape

    f_i = capture_histories.sum(axis=1)
    p_hat = np.clip(float(np.mean(f_i)) / D, 0.01, 0.99)

    # Binomial probabilities
    b = np.zeros(D + 1)
    for k_val in range(D + 1):
        b[k_val] = comb(D, k_val) * (p_hat ** k_val) * ((1 - p_hat) ** (D - k_val))

    # Observed counts
    O = np.array([np.sum(f_i == k) for k in range(1, D + 1)], dtype=float)

    def chi2(N_val):
        if N_val < n_obs or N_val > n_obs * 50:
            return 1e20
        stat = 0.0
        # Compare capture frequency distribution
        for k in range(1, D + 1):
            expected = N_val * b[k]
            if expected < 0.5:
                continue
            stat += (O[k - 1] - expected) ** 2 / expected
        # Match total observed: n_obs ≈ N * (1 - b_0)
        expected_n = N_val * (1.0 - b[0])
        stat += (n_obs - expected_n) ** 2 / max(expected_n, 1.0)
        return stat

    res = minimize_scalar(chi2, bounds=(n_obs, n_obs * 20), method='bounded')
    return float(max(res.x, n_obs))


def min_chisq_cr_improved(
    capture_histories: np.ndarray,
    K: int = 5,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
    ci_method: str = 'asymptotic',
    verbose: bool = False,
) -> dict:
    """
    Improved minimum chi-square CR estimator with bootstrap CI.

    The chi-square objective compares the observed capture-frequency
    distribution to the expected binomial distribution, minimising over N.

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    K : number of groups (used for adaptive grouping fallback only)
    alpha : significance level
    n_boot : bootstrap replicates (for smoothed/parametric_bc methods)
    seed : random seed
    ci_method : 'asymptotic' (default, calibrated), 'smoothed', 'parametric_bc', 'jackknife'
    verbose : show progress bar

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, etc.
    """
    n_obs, D = capture_histories.shape

    # Fallback for degenerate cases
    f_i = capture_histories.sum(axis=1)
    if len(np.unique(f_i)) < 2:
        from .estimators import _closed_cr_point
        N_fallback = _closed_cr_point(capture_histories)
        return {
            'N_hat': N_fallback,
            'variance': 0.0,
            'ci_lower': float(n_obs),
            'ci_upper': float(N_fallback),
            'method': 'min_chisq_fallback',
            'ci_method': ci_method,
            'calibration_factor': 1.0,
            'n_observed': n_obs,
            'n_ibnr': max(N_fallback - n_obs, 0),
        }

    # Estimate overdispersion for smoothed bootstrap
    p_mean = float(np.mean(f_i)) / D
    var_binom = D * p_mean * (1.0 - p_mean)
    var_obs = float(np.var(f_i, ddof=1))
    c_hat = max(1.0, var_obs / max(var_binom, 0.001))

    if ci_method == 'smoothed':
        result = smoothed_frequency_bootstrap_ci(
            _min_chisq_improved_point, capture_histories, K,
            n_boot=n_boot, alpha=alpha, seed=seed,
            lower_bound=float(n_obs), overdispersion=c_hat,
            verbose=verbose,
        )
        se_key = 'se_boot'
    elif ci_method == 'parametric_bc':
        result = parametric_bootstrap_bc(
            _min_chisq_improved_point, capture_histories, K,
            n_boot=n_boot, alpha=alpha, seed=seed,
            lower_bound=float(n_obs), verbose=verbose,
        )
        se_key = 'se_boot'
    elif ci_method == 'asymptotic':
        result = asymptotic_sandwich_ci(
            _min_chisq_improved_point, capture_histories, K,
            alpha=alpha, lower_bound=float(n_obs),
            estimator_type='min_chisq',
            calibrate=True, n_calib=200, seed=seed, verbose=verbose,
        )
        se_key = 'se_asymptotic'
    elif ci_method == 'jackknife':
        result = jackknife_sandwich_ci(
            _min_chisq_improved_point, capture_histories, K,
            alpha=alpha, lower_bound=float(n_obs),
        )
        se_key = 'se_sandwich'
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")

    return {
        'N_hat': result['N_hat'],
        'variance': result[se_key] ** 2,
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'method': 'min_chisq_improved',
        'ci_method': ci_method,
        'calibration_factor': result.get('calibration_factor', 1.0),
        'K_actual': K,
        'n_observed': n_obs,
        'n_ibnr': max(result['N_hat'] - n_obs, 0),
        'se_boot': result[se_key],
        'n_failed': result.get('n_failed', 0),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_pi_logistic(
    X: np.ndarray, covariates: Optional[np.ndarray]
) -> np.ndarray:
    """Estimate per-period detection probabilities via logistic regression."""
    n_obs, D = X.shape

    if covariates is not None:
        from sklearn.linear_model import LogisticRegression
        p_hat = np.zeros((n_obs, D))
        for d in range(D):
            y_d = X[:, d]
            if len(np.unique(y_d)) < 2:
                p_hat[:, d] = y_d.mean()
                continue
            try:
                clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
                clf.fit(covariates, y_d)
                p_hat[:, d] = clf.predict_proba(covariates)[:, 1]
            except Exception:
                p_hat[:, d] = y_d.mean()
        return np.clip(p_hat, 0.01, 0.99)
    else:
        p_col = np.clip(X.mean(axis=0), 0.01, 0.99)
        return np.tile(p_col, (n_obs, 1))


def _estimate_pi_rf(
    X: np.ndarray, covariates: Optional[np.ndarray]
) -> np.ndarray:
    """Estimate detection probabilities via random forest."""
    if covariates is None:
        return _estimate_pi_logistic(X, None)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return _estimate_pi_logistic(X, covariates)

    n_obs, D = X.shape
    p_hat = np.zeros((n_obs, D))
    for d in range(D):
        y_d = X[:, d]
        if len(np.unique(y_d)) < 2:
            p_hat[:, d] = y_d.mean()
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(covariates, y_d)
        p_hat[:, d] = rf.predict_proba(covariates)[:, 1]

    return np.clip(p_hat, 0.01, 0.99)


def _estimate_p0(pi_hat: np.ndarray, D: int) -> np.ndarray:
    """Estimate P(never observed) from per-period detection probabilities."""
    p0 = np.ones(pi_hat.shape[0])
    for d in range(min(D, pi_hat.shape[1])):
        p0 *= (1.0 - pi_hat[:, d])
    return np.clip(p0, 0.001, 0.999)
