"""
Variance estimation for CR estimators.
"""

import numpy as np
from typing import Optional


def delta_method_variance(Y_arr: np.ndarray, x_arr: np.ndarray, N_hat: float) -> float:
    """
    Delta method variance for N_hat = sum(Y)/sum(x).

    Uses the multivariate delta method on g(S_x, S_Y) = S_Y / S_x.
    """
    m = len(Y_arr)
    if m == 0 or N_hat <= 0:
        return 0.0

    q = Y_arr / max(N_hat, 1)
    q = np.clip(q, 1e-10, 1 - 1e-10)

    var_sx = np.sum(q * (1 - q))

    var_sy = 0.0
    for i in range(m):
        for k in range(m):
            min_ik = min(i, k)
            cov_ik = np.sum(q[:min_ik + 1] * (1 - q[:min_ik + 1]))
            var_sy += cov_ik

    sum_q = q.sum()
    if sum_q <= 0:
        return float('inf')

    g_sx = -N_hat / (sum_q)
    g_sy = 1.0 / sum_q

    cov_sx_sy = -np.sum((np.arange(m, 0, -1)) * q * (1 - q))

    var_N = g_sx ** 2 * var_sx + g_sy ** 2 * var_sy + 2 * g_sx * g_sy * cov_sx_sy

    return max(var_N, 0)


def bootstrap_variance(capture_histories: np.ndarray, B: int = 200,
                        estimator_fn=None) -> dict:
    """
    Nonparametric bootstrap variance estimate.

    Parameters
    ----------
    capture_histories : np.ndarray, shape (n, D)
    B : int
        Number of bootstrap replicates.
    estimator_fn : callable
        Function that takes capture_histories and returns CRResult.
    """
    from .estimators import closed_system_cr

    if estimator_fn is None:
        estimator_fn = closed_system_cr

    n, D = capture_histories.shape
    boot_estimates = []

    for b in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = capture_histories[idx]
        result = estimator_fn(boot_data)
        boot_estimates.append(result.N_hat)

    boot_arr = np.array(boot_estimates)

    return {
        'var_boot': np.var(boot_arr, ddof=1),
        'se_boot': np.std(boot_arr, ddof=1),
        'ci_boot_lower': np.percentile(boot_arr, 2.5),
        'ci_boot_upper': np.percentile(boot_arr, 97.5),
        'boot_mean': np.mean(boot_arr),
        'boot_median': np.median(boot_arr),
    }
