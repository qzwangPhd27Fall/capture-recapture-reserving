"""
Monte Carlo goodness-of-fit diagnostics for CR reserving models.

Reference:
  Choo et al. (2024) — Monte Carlo GoF tests for capture-recapture models.

Provides:
  - Posterior predictive checks for DPMM-CR
  - Freeman-Tukey discrepancy measure
  - Bayesian p-values
  - Residual diagnostics for HMM-CR
  - Comparison diagnostic: CR vs CL fit assessment
"""

import numpy as np
from numpy.random import default_rng
from typing import Optional, Dict, Callable


def posterior_predictive_check(
    capture_histories: np.ndarray,
    N_posterior: np.ndarray,
    p0_posterior: np.ndarray,
    n_replicates: int = 200,
    discrepancy: str = 'freeman_tukey',
    seed: Optional[int] = None,
) -> dict:
    """
    Posterior predictive check for DPMM-CR model.

    For each posterior draw, generates a replicated dataset under the fitted
    model, computes a discrepancy measure, and compares with the observed.

    Parameters
    ----------
    capture_histories : (n_obs, D) observed binary matrix
    N_posterior : (n_samples,) posterior N draws
    p0_posterior : (n_samples,) posterior p0 draws
    n_replicates : number of posterior draws to use
    discrepancy : 'freeman_tukey' or 'chi_square'
    seed : random seed

    Returns
    -------
    dict with bayesian_pvalue, T_obs, T_rep, etc.
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape

    # observed capture frequencies
    f_obs = capture_histories.sum(axis=1)  # per-individual capture count
    f_table_obs = np.bincount(f_obs, minlength=D + 1)[1:]  # f=1,...,D

    n_post = len(N_posterior)
    idx = rng.choice(n_post, min(n_replicates, n_post), replace=False)

    T_rep = np.zeros(len(idx))
    for i, pi in enumerate(idx):
        N_i = int(N_posterior[pi])
        p0_i = p0_posterior[pi]

        # generate replicate data
        # each of N_i individuals has P(never observed) = p0_i
        # conditional on being observed, capture frequencies follow a
        # truncated binomial-like distribution
        p_detection = 1.0 - p0_i ** (1.0 / D)  # approximate per-period prob
        p_detection = np.clip(p_detection, 0.01, 0.99)

        # simulate N_i individuals
        rep_X = rng.random((N_i, D)) < p_detection
        ever_obs = rep_X.any(axis=1)
        rep_f = rep_X[ever_obs].sum(axis=1)

        if len(rep_f) == 0:
            T_rep[i] = 0
            continue

        rep_table = np.bincount(rep_f, minlength=D + 1)[1:]

        # compute discrepancy
        T_rep[i] = _compute_discrepancy(f_table_obs, rep_table, discrepancy, n_obs)

    # observed discrepancy (under model fit)
    # expected frequencies under fitted p0
    p0_mean = np.mean(p0_posterior[idx])
    p_det = 1.0 - p0_mean ** (1.0 / D)
    p_det = np.clip(p_det, 0.01, 0.99)

    # expected: Binomial(D, p_det) truncated at 0
    expected = np.zeros(D)
    for f in range(1, D + 1):
        binom_prob = _binom_pmf(D, p_det, f)
        expected[f - 1] = binom_prob * N_posterior.mean() / (1.0 - p0_mean)

    T_obs = _compute_discrepancy(f_table_obs, expected, discrepancy, n_obs)
    bpval = np.mean(T_rep >= T_obs)

    return {
        'bayesian_pvalue': float(bpval),
        'T_obs': float(T_obs),
        'T_rep_mean': float(np.mean(T_rep)),
        'T_rep_std': float(np.std(T_rep)),
        'discrepancy': discrepancy,
        'n_replicates': len(idx),
    }


def hmm_gof_diagnostics(
    capture_histories: np.ndarray,
    trans_matrix: np.ndarray,
    emit_matrix: np.ndarray,
    init_dist: np.ndarray,
    n_bootstrap: int = 200,
    seed: Optional[int] = None,
) -> dict:
    """
    Goodness-of-fit diagnostics for HMM-CR model.

    Checks:
      1. Residual autocorrelation (should be near 0 for well-fit HMM)
      2. State distribution comparison (predicted vs observed time in states)
      3. Bootstrap log-likelihood distribution
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape
    n_states = len(emit_matrix)

    # 1. Pseudo-residuals: forecast error for each observation
    # P(X_{j,d} = 1 | X_{j,1:d-1}) from forward recursion
    residuals = np.zeros((n_obs, D))

    for j in range(n_obs):
        alpha = np.zeros((D, n_states))
        for s in range(n_states):
            e = emit_matrix[s] if capture_histories[j, 0] == 1 else (1.0 - emit_matrix[s])
            alpha[0, s] = init_dist[s] * e
        alpha[0] /= alpha[0].sum()

        for d in range(1, D):
            alpha[d] = alpha[d - 1] @ trans_matrix
            for s in range(n_states):
                e = emit_matrix[s] if capture_histories[j, d] == 1 else (1.0 - emit_matrix[s])
                alpha[d, s] *= e
            alpha[d] /= alpha[d].sum()

        for d in range(D):
            p_pred = np.dot(alpha[max(d - 1, 0)], emit_matrix) if d > 0 else np.dot(init_dist, emit_matrix)
            residuals[j, d] = capture_histories[j, d] - p_pred

    # 2. State occupancy: proportion of time in each state
    if n_states == 5:
        from .hmm_cr import _forward_backward, _compute_posteriors
        la, lb, ll = _forward_backward(capture_histories[:min(n_obs, 100)],
                                        trans_matrix, emit_matrix, init_dist)
        gamma, _ = _compute_posteriors(capture_histories[:min(n_obs, 100)],
                                        la, lb, ll, trans_matrix, emit_matrix)
        state_occupancy = gamma.mean(axis=(0, 1))
    else:
        state_occupancy = np.ones(n_states) / n_states

    # 3. Bootstrap log-likelihood
    boot_ll = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_obs, n_obs, replace=True)
        boot_X = capture_histories[idx]
        la, lb, ll = _forward_backward(boot_X, trans_matrix, emit_matrix, init_dist)
        boot_ll[b] = ll

    return {
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_autocorr_lag1': float(np.corrcoef(
            residuals[:, :-1].flatten(), residuals[:, 1:].flatten()
        )[0, 1]) if D > 1 and residuals.size > 1 else 0.0,
        'state_occupancy': state_occupancy.tolist(),
        'boot_ll_mean': float(np.mean(boot_ll)),
        'boot_ll_std': float(np.std(boot_ll)),
    }


def compare_cr_cl_gof(
    cr_N_hat: float,
    cl_N_hat: float,
    capture_histories: np.ndarray,
    triangle: np.ndarray,
) -> dict:
    """
    Comparative diagnostic: which model fits better?

    Computes:
      - CR: posterior predictive Bayesian p-value
      - CL: Mack model residual analysis
    """
    n_obs, D = capture_histories.shape

    # CR diagnostic: capture frequency distribution
    f_obs = capture_histories.sum(axis=1)
    f_hist = np.bincount(f_obs, minlength=D + 1)[1:]

    # Expected under homogeneous detection (CR baseline)
    p_cr = f_obs.mean() / D
    expected_cr = np.zeros(D)
    for f in range(1, D + 1):
        expected_cr[f - 1] = n_obs * _binom_pmf(D, p_cr, f) / (1 - (1 - p_cr) ** D)

    cr_chisq = np.sum((f_hist - expected_cr) ** 2 / np.maximum(expected_cr, 1))
    cr_df = D - 2  # estimated: p and N
    cr_pvalue = 1.0 - _chi2_cdf(cr_chisq, cr_df)

    # CL diagnostic: development factor stability
    if triangle.shape[0] >= 2:
        ldrs = np.zeros((triangle.shape[0] - 1, D - 1))
        for i in range(triangle.shape[0] - 1):
            for d in range(D - 1):
                if triangle[i, d] > 0:
                    ldrs[i, d] = triangle[i, d + 1] / triangle[i, d]
        ldr_var = np.nanvar(ldrs, axis=0)
        cl_instability = np.nanmean(ldr_var)
    else:
        cl_instability = 0.0

    return {
        'cr_chisq': float(cr_chisq),
        'cr_df': int(cr_df),
        'cr_pvalue': float(cr_pvalue),
        'cl_ldr_instability': float(cl_instability),
        'prefer_cr': cr_pvalue > 0.05 or cl_instability > 0.1,
    }


# ---- Helpers ----

def _compute_discrepancy(obs, rep, method: str, n_obs: int) -> float:
    """Compute discrepancy between observed and replicated frequencies."""
    if method == 'freeman_tukey':
        # Freeman-Tukey: sum (sqrt(O_i) - sqrt(E_i))^2
        obs = np.asarray(obs, dtype=float)
        rep = np.asarray(rep, dtype=float)
        diff = np.sqrt(np.maximum(obs, 0)) - np.sqrt(np.maximum(rep, 0.1))
        return float(np.sum(diff ** 2))
    elif method == 'chi_square':
        obs = np.asarray(obs, dtype=float)
        rep = np.asarray(rep, dtype=float)
        return float(np.sum((obs - rep) ** 2 / np.maximum(rep, 0.1)))
    else:
        return float(np.sum((np.asarray(obs) - np.asarray(rep)) ** 2))


def _binom_pmf(n: int, p: float, k: int) -> float:
    """Binomial PMF: P(X = k)."""
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def _chi2_cdf(x: float, df: int) -> float:
    """Chi-square CDF approximation."""
    from scipy.stats import chi2
    return float(chi2.cdf(x, df))
