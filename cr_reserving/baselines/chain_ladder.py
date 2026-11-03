"""
Chain-Ladder and Bornhuetter-Ferguson baselines.

Implements standard actuarial reserving methods for comparison with CR.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CLResult:
    ultimate: np.ndarray
    reserves: np.ndarray
    ibnr_counts: Optional[np.ndarray]
    ldfs: np.ndarray
    mack_se: Optional[np.ndarray]
    method: str
    details: dict


def chain_ladder(triangle: np.ndarray, alpha: float = 0.05) -> CLResult:
    """
    Standard Chain-Ladder method with Mack variance.

    Parameters
    ----------
    triangle : np.ndarray, shape (A, D)
        Cumulative loss triangle. triangle[a, d] = cumulative paid/incurred
        for accident year a at development d. Use np.nan for unknown entries.
    alpha : float
        Significance level for CI.

    Returns
    -------
    CLResult with ultimates, reserves, LDFs, and Mack standard errors.
    """
    A, D = triangle.shape

    ldfs = np.zeros(D - 1)
    sigma2 = np.zeros(D - 1)

    for d in range(D - 1):
        num = 0.0
        den = 0.0
        for a in range(A):
            if not np.isnan(triangle[a, d]) and not np.isnan(triangle[a, d + 1]):
                num += triangle[a, d + 1]
                den += triangle[a, d]
        ldfs[d] = num / den if den > 0 else 1.0

        ss = 0.0
        count = 0
        for a in range(A):
            if not np.isnan(triangle[a, d]) and not np.isnan(triangle[a, d + 1]):
                ratio = triangle[a, d + 1] / triangle[a, d] if triangle[a, d] > 0 else ldfs[d]
                ss += triangle[a, d] * (ratio - ldfs[d]) ** 2
                count += 1
        sigma2[d] = ss / max(count - 1, 1) if count > 1 else 0

    ultimate = np.zeros(A)
    for a in range(A):
        latest_d = -1
        for d in range(D):
            if not np.isnan(triangle[a, d]):
                latest_d = d

        if latest_d < 0:
            ultimate[a] = 0
            continue

        cum_ldf = 1.0
        for d in range(latest_d, D - 1):
            cum_ldf *= ldfs[d]
        ultimate[a] = triangle[a, latest_d] * cum_ldf

    reserves = np.zeros(A)
    for a in range(A):
        latest_d = -1
        for d in range(D):
            if not np.isnan(triangle[a, d]):
                latest_d = d
        if latest_d >= 0:
            reserves[a] = ultimate[a] - triangle[a, latest_d]

    mack_se = _mack_standard_error(triangle, ldfs, sigma2, ultimate)

    return CLResult(
        ultimate=ultimate,
        reserves=reserves,
        ibnr_counts=None,
        ldfs=ldfs,
        mack_se=mack_se,
        method='chain_ladder',
        details={'sigma2': sigma2.tolist(), 'total_reserve': reserves.sum()}
    )


def bornhuetter_ferguson(triangle: np.ndarray,
                         prior_ultimate: np.ndarray,
                         alpha: float = 0.05) -> CLResult:
    """
    Bornhuetter-Ferguson method.

    Parameters
    ----------
    triangle : np.ndarray, shape (A, D)
        Cumulative loss triangle.
    prior_ultimate : np.ndarray, shape (A,)
        A priori ultimate loss estimates (e.g., from premium * expected loss ratio).
    """
    A, D = triangle.shape

    cl_result = chain_ladder(triangle, alpha)
    ldfs = cl_result.ldfs

    cum_ldfs = np.ones(D)
    for d in range(D - 2, -1, -1):
        cum_ldfs[d] = cum_ldfs[d + 1] * ldfs[d] if d < D - 1 else ldfs[d]
        if d < D - 1:
            cum_ldfs[d] = np.prod(ldfs[d:])

    development_pct = np.zeros(D)
    for d in range(D):
        tail_ldf = np.prod(ldfs[d:]) if d < D - 1 else 1.0
        development_pct[d] = 1.0 / tail_ldf

    bf_ultimate = np.zeros(A)
    bf_reserves = np.zeros(A)

    for a in range(A):
        latest_d = -1
        for d in range(D):
            if not np.isnan(triangle[a, d]):
                latest_d = d

        if latest_d < 0:
            bf_ultimate[a] = prior_ultimate[a]
            bf_reserves[a] = prior_ultimate[a]
            continue

        pct_developed = development_pct[latest_d]
        pct_unreported = 1 - pct_developed

        bf_reserves[a] = prior_ultimate[a] * pct_unreported
        bf_ultimate[a] = triangle[a, latest_d] + bf_reserves[a]

    return CLResult(
        ultimate=bf_ultimate,
        reserves=bf_reserves,
        ibnr_counts=None,
        ldfs=ldfs,
        mack_se=None,
        method='bornhuetter_ferguson',
        details={'prior_ultimate': prior_ultimate.tolist(),
                 'total_reserve': bf_reserves.sum()}
    )


def _mack_standard_error(triangle, ldfs, sigma2, ultimate):
    """Mack (1993) standard error for CL reserves."""
    A, D = triangle.shape
    mack_se = np.zeros(A)

    for a in range(A):
        latest_d = -1
        for d in range(D):
            if not np.isnan(triangle[a, d]):
                latest_d = d

        if latest_d >= D - 1:
            mack_se[a] = 0
            continue

        var_sum = 0.0
        for d in range(latest_d, D - 1):
            if sigma2[d] <= 0:
                continue

            C_ad = triangle[a, latest_d] * np.prod(ldfs[latest_d:d])
            if C_ad <= 0:
                continue

            sum_C = 0.0
            for a2 in range(A):
                if not np.isnan(triangle[a2, d]):
                    sum_C += triangle[a2, d]

            term = sigma2[d] / (ldfs[d] ** 2) * (1.0 / C_ad + 1.0 / max(sum_C, 1))
            var_sum += term

        mack_se[a] = ultimate[a] * np.sqrt(max(var_sum, 0))

    return mack_se


def individual_to_triangle(capture_histories: np.ndarray,
                           severities: Optional[np.ndarray] = None,
                           accident_years: Optional[np.ndarray] = None,
                           n_accident_years: int = 10,
                           n_dev_periods: int = 10) -> np.ndarray:
    """
    Convert individual-level claim data to an aggregate run-off triangle.

    This is the bridge function: takes CR-format data and creates the
    CL-format triangle for baseline comparison.

    Parameters
    ----------
    capture_histories : np.ndarray, shape (n_claims, D)
    severities : np.ndarray, shape (n_claims,), optional
        Claim amounts. If None, uses claim counts.
    accident_years : np.ndarray, shape (n_claims,), optional
        Accident year assignment. If None, assigns uniformly.
    n_accident_years : int
    n_dev_periods : int

    Returns
    -------
    triangle : np.ndarray, shape (n_accident_years, n_dev_periods)
        Cumulative triangle (count or amount).
    """
    n_claims, D_orig = capture_histories.shape

    if accident_years is None:
        accident_years = np.random.randint(0, n_accident_years, n_claims)
    if severities is None:
        severities = np.ones(n_claims)

    D = min(D_orig, n_dev_periods)
    triangle = np.full((n_accident_years, D), np.nan)

    for a in range(n_accident_years):
        mask = accident_years == a
        if not np.any(mask):
            continue

        claims_a = capture_histories[mask, :D]
        sev_a = severities[mask]

        max_dev = D - a
        if max_dev <= 0:
            continue

        for d in range(min(max_dev, D)):
            cum_count = 0
            for j in range(len(claims_a)):
                if np.any(claims_a[j, :d + 1] == 1):
                    cum_count += sev_a[j]
            triangle[a, d] = cum_count

    return triangle
