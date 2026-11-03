"""
Core capture-recapture estimators for insurance claims reserving.

Implements:
  - Closed-system CR estimator (Rehman et al. 2016)
  - Open-system CR estimator with entry/exit dynamics
  - Minimum chi-square CR estimator (Zheng et al. 2023)
  - HMM-based CR estimator (state-transition model)

All methods use nonparametric bootstrap for CI (B=500, percentile).
"""

import numpy as np
from math import comb
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize_scalar

from .bootstrap import bootstrap_ci, parametric_bootstrap_cr


@dataclass
class CRResult:
    N_hat: float
    variance: float
    ci_lower: float
    ci_upper: float
    method: str
    details: dict


# ---------------------------------------------------------------------------
# Point estimators (no CI, just return N_hat float)
# These are the bootstrap building blocks.
# ---------------------------------------------------------------------------

def _closed_cr_point(capture_histories: np.ndarray) -> float:
    """
    Closed-system passive CR point estimate — Schnabel estimator.

    N_hat = Σ_d (n_d · M_d) / Σ_d R_d

    where for each period d:
      n_d = distinct individuals marked BEFORE period d
      M_d = total captures in period d
      R_d = recaptures in period d (already-marked individuals)
    """
    n, D = capture_histories.shape

    marked = set()
    numerator = 0.0
    denominator = 0.0

    for d in range(D):
        # Individuals captured in period d
        captured_d = np.where(capture_histories[:, d] == 1)[0]
        M_d = len(captured_d)
        if M_d == 0:
            continue

        n_d = len(marked)  # marked before this period
        R_d = sum(1 for j in captured_d if j in marked)

        numerator += n_d * M_d
        denominator += R_d

        # Mark all individuals captured in this period
        for j in captured_d:
            marked.add(j)

    if denominator == 0:
        return float(n)

    N_hat = numerator / denominator
    return max(N_hat, float(n))


def _open_cr_point(
    capture_histories: np.ndarray,
    entry_times: Optional[np.ndarray] = None,
    exit_times: Optional[np.ndarray] = None,
) -> float:
    """
    Open-system CR point estimate.

    Y_i = Y_{i-1} + (1 - x_i) - s_i + k_i
    where s_i are exits and k_i are entries between events.
    """
    n, D = capture_histories.shape

    if entry_times is None:
        entry_times = np.zeros(n, dtype=int)
    if exit_times is None:
        exit_times = np.full(n, D + 1, dtype=int)

    # Fast path: closed system (all entries at 0, all exits after D)
    is_closed = bool(np.all(entry_times == 0) and np.all(exit_times > D))

    events = []
    for d in range(D):
        active_mask = (entry_times <= d) & (exit_times > d)
        active_claims = np.where(capture_histories[:, d] == 1)[0]
        for j in active_claims:
            if active_mask[j]:
                events.append((j, d))

    if not events:
        return float(n)

    seen = set()
    Y_vals = []
    x_vals = []

    if is_closed:
        # All marked individuals always active
        for claim_id, period in events:
            is_recapture = 1 if claim_id in seen else 0
            Y_vals.append(len(seen))
            x_vals.append(is_recapture)
            seen.add(claim_id)
    else:
        # Open system: track active marked with precomputed arrays
        entry_arr = np.asarray(entry_times)
        exit_arr = np.asarray(exit_times)
        seen_list = []  # list of claim_ids for indexing
        for claim_id, period in events:
            is_recapture = 1 if claim_id in seen else 0
            # Count active marked using numpy boolean indexing
            if seen_list:
                ids = np.array(seen_list)
                active = (entry_arr[ids] <= period) & (exit_arr[ids] > period)
                active_marked = int(active.sum())
            else:
                active_marked = 0
            Y_vals.append(active_marked)
            x_vals.append(is_recapture)
            seen.add(claim_id)
            seen_list.append(claim_id)

    Y_arr = np.array(Y_vals, dtype=float)
    x_arr = np.array(x_vals, dtype=float)

    sum_x = x_arr.sum()
    if sum_x == 0:
        return float(n)

    N_hat = Y_arr.sum() / sum_x
    return max(N_hat, n)


def _min_chisq_point(capture_histories: np.ndarray, K: int = 5) -> float:
    """
    Minimum chi-square CR point estimate.

    Partitions capture events into K groups and finds N that
    minimizes the Pearson chi-square between observed and
    expected capture frequencies.

    χ²(N) = Σ_{k=1}^D (O_k - N·b_k)² / (N·b_k)

    where O_k = count of individuals with exactly k captures,
    b_k = Binom(D, p̂, k), p̂ = f̄ / D.
    """
    n, D = capture_histories.shape

    # Capture frequency distribution
    f_i = capture_histories.sum(axis=1)
    p_hat = np.clip(f_i.mean() / D, 0.01, 0.99)

    # Binomial probabilities
    b = np.zeros(D + 1)
    for k in range(D + 1):
        b[k] = comb(D, k) * (p_hat ** k) * ((1 - p_hat) ** (D - k))

    # Observed counts per capture frequency (k=1,...,D)
    O = np.array([np.sum(f_i == k) for k in range(1, D + 1)], dtype=float)

    # χ²(N) = Σ (O_k - N·b_k)² / (N·b_k) for k ≥ 1
    def chi2(N_val):
        if N_val < n:
            return 1e20
        stat = 0.0
        for k in range(1, D + 1):
            expected = N_val * b[k]
            if expected < 0.5:
                # Merge small cells: add to adjacent
                continue
            stat += (O[k - 1] - expected) ** 2 / expected
        # Penalise N that predicts too few observed: n_obs ≈ N * (1-b_0)
        expected_n = N_val * (1.0 - b[0])
        stat += (n - expected_n) ** 2 / max(expected_n, 1.0)
        return stat

    res = minimize_scalar(chi2, bounds=(n, n * 20), method='bounded')
    N_hat = max(res.x, n)

    return float(N_hat)


# ---------------------------------------------------------------------------
# Full estimators with bootstrap CI
# ---------------------------------------------------------------------------

def closed_system_cr(
    capture_histories: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> CRResult:
    """
    Closed-system passive CR estimator with bootstrap CI.

    N_hat = ΣY_i / Σx_i

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    alpha : significance level
    n_boot : bootstrap replicates
    seed : random seed
    """
    n, D = capture_histories.shape

    result = parametric_bootstrap_cr(
        _closed_cr_point, capture_histories,
        n_boot=n_boot, alpha=alpha, seed=seed,
        lower_bound=float(n),
    )

    return CRResult(
        N_hat=result['N_hat'],
        variance=result['se_boot'] ** 2,
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        method='closed_cr',
        details={
            'n_observed': n,
            'n_ibnr': max(result['N_hat'] - n, 0),
            'se_boot': result['se_boot'],
            'n_boot': n_boot,
            'n_failed': result['n_failed'],
        }
    )


def open_system_cr(
    capture_histories: np.ndarray,
    entry_times: Optional[np.ndarray] = None,
    exit_times: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> CRResult:
    """
    Open-system CR estimator with entry/exit dynamics and bootstrap CI.

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    entry_times : (n_obs,) entry period for each observed claim
    exit_times : (n_obs,) exit period for each observed claim
    alpha : significance level
    n_boot : bootstrap replicates
    seed : random seed
    """
    n, D = capture_histories.shape

    if entry_times is None:
        entry_times = np.zeros(n, dtype=int)
    if exit_times is None:
        exit_times = np.full(n, D + 1, dtype=int)

    result = bootstrap_ci(
        _open_cr_point, capture_histories, entry_times, exit_times,
        n_boot=n_boot, alpha=alpha, seed=seed,
        lower_bound=float(n),
    )

    return CRResult(
        N_hat=result['N_hat'],
        variance=result['se_boot'] ** 2,
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        method='open_cr',
        details={
            'n_observed': n,
            'n_ibnr': max(result['N_hat'] - n, 0),
            'se_boot': result['se_boot'],
            'n_boot': n_boot,
            'n_failed': result['n_failed'],
        }
    )


def min_chisq_cr(
    capture_histories: np.ndarray,
    K: int = 5,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> CRResult:
    """
    Minimum chi-square CR estimator with bootstrap CI.

    Finds N that minimises χ²(N) = Σ (O_k - E_k(N))² / E_k(N)
    where O_k is the observed count of individuals with exactly k captures
    and E_k(N) = N · Binom(D, p̂, k).

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    K : not used (kept for API compatibility), method uses D categories
    alpha : significance level
    n_boot : bootstrap replicates
    seed : random seed
    """
    n, D = capture_histories.shape

    result = parametric_bootstrap_cr(
        _min_chisq_point, capture_histories, K,
        n_boot=n_boot, alpha=alpha, seed=seed,
        lower_bound=float(n),
    )

    return CRResult(
        N_hat=result['N_hat'],
        variance=result['se_boot'] ** 2,
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        method='min_chisq_cr',
        details={
            'K': K,
            'n_observed': n,
            'n_ibnr': max(result['N_hat'] - n, 0),
            'se_boot': result['se_boot'],
            'n_boot': n_boot,
            'n_failed': result['n_failed'],
        }
    )


def hmm_cr(
    capture_histories: np.ndarray,
    n_states: int = 4,
    max_iter: int = 100,
    alpha: float = 0.05,
) -> CRResult:
    """
    HMM-based CR estimator using EM algorithm (simplified, 4-state).

    This is a lightweight version. For full Baum-Welch with 5 states,
    use hmm_cr_baum_welch from hmm_cr.py.

    Parameters
    ----------
    capture_histories : (n_observed, D) binary matrix
    n_states : number of HMM states (default 4: U, R, A, S)
    max_iter : maximum EM iterations
    alpha : significance level
    """
    n, D = capture_histories.shape

    # Initialise transition matrix
    trans = np.zeros((n_states, n_states))
    trans[0, :] = 0
    trans[0, 0] = 0.7
    trans[0, 1] = 0.3
    trans[n_states - 1, :] = 0
    trans[n_states - 1, n_states - 1] = 1.0
    for i in range(1, n_states - 1):
        trans[i, :] = 0
        trans[i, i] = 0.4
        if i + 1 < n_states:
            trans[i, i + 1] = 0.6

    emit = np.array([0.0, 0.8, 0.9, 0.1])
    init_dist = np.array([0.5, 0.3, 0.15, 0.05])

    for iteration in range(max_iter):
        emit_num = np.zeros(n_states)
        emit_den = np.zeros(n_states)

        for j in range(n):
            h = capture_histories[j]

            log_alpha = np.full((D, n_states), -np.inf)
            for s in range(n_states):
                e = emit[s] if h[0] == 1 else (1 - emit[s])
                if init_dist[s] > 0 and e > 0:
                    log_alpha[0, s] = np.log(init_dist[s]) + np.log(e)

            for d in range(1, D):
                for s in range(n_states):
                    e = emit[s] if h[d] == 1 else (1 - emit[s])
                    if e <= 0:
                        continue
                    vals = []
                    for s_prev in range(n_states):
                        if trans[s_prev, s] > 0 and log_alpha[d-1, s_prev] > -1e300:
                            vals.append(log_alpha[d-1, s_prev] + np.log(trans[s_prev, s]))
                    if vals:
                        m = max(vals)
                        log_alpha[d, s] = m + np.log(sum(np.exp(v - m) for v in vals)) + np.log(e)

            log_beta = np.full((D, n_states), -np.inf)
            log_beta[D - 1, :] = 0.0
            for d in range(D - 2, -1, -1):
                for s in range(n_states):
                    vals = []
                    for s_next in range(n_states):
                        e = emit[s_next] if h[d+1] == 1 else (1 - emit[s_next])
                        if trans[s, s_next] > 0 and e > 0 and log_beta[d+1, s_next] > -1e300:
                            vals.append(np.log(trans[s, s_next]) + np.log(e) + log_beta[d+1, s_next])
                    if vals:
                        m = max(vals)
                        log_beta[d, s] = m + np.log(sum(np.exp(v - m) for v in vals))

            log_evidence_vals = log_alpha[D-1, :]
            max_ev = np.max(log_evidence_vals)
            if max_ev > -1e300:
                log_evidence = max_ev + np.log(sum(np.exp(log_evidence_vals - max_ev)))

                for d in range(D):
                    log_gamma = log_alpha[d] + log_beta[d] - log_evidence
                    gamma = np.exp(np.clip(log_gamma, -700, 0))
                    emit_num += gamma * h[d]
                    emit_den += gamma

        new_emit = np.where(emit_den > 0, emit_num / emit_den, emit)
        new_emit[0] = 0.0
        new_emit = np.clip(new_emit, 0.01, 0.99)
        new_emit[0] = 0.0

        if np.max(np.abs(new_emit - emit)) < 1e-6:
            emit = new_emit
            break
        emit = new_emit

    p0 = _estimate_p0_hmm(init_dist, trans, emit, D)
    p0 = max(p0, 0.01)
    N_hat = max(n / (1 - p0), n)

    # Bootstrap CI for HMM
    n_boot = 200
    rng = np.random.default_rng(None)
    boot_N = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_hist = capture_histories[idx]
        try:
            bt = trans.copy()
            be = emit.copy()
            bi = init_dist.copy()
            for it in range(30):
                be_num = np.zeros(n_states)
                be_den = np.zeros(n_states)
                for jb in range(len(boot_hist)):
                    hb = boot_hist[jb]
                    la = np.full((D, n_states), -np.inf)
                    for s in range(n_states):
                        e = be[s] if hb[0] == 1 else (1 - be[s])
                        if bi[s] > 0 and e > 0:
                            la[0, s] = np.log(bi[s]) + np.log(e)
                    for d in range(1, D):
                        for s in range(n_states):
                            e = be[s] if hb[d] == 1 else (1 - be[s])
                            if e <= 0:
                                continue
                            vals = []
                            for sp in range(n_states):
                                if bt[sp, s] > 0 and la[d-1, sp] > -1e300:
                                    vals.append(la[d-1, sp] + np.log(bt[sp, s]))
                            if vals:
                                m = max(vals)
                                la[d, s] = m + np.log(sum(np.exp(v - m) for v in vals)) + np.log(e)
                    lb = np.full((D, n_states), -np.inf)
                    lb[D-1, :] = 0.0
                    for d in range(D-2, -1, -1):
                        for s in range(n_states):
                            vals = []
                            for sn in range(n_states):
                                e = be[sn] if hb[d+1] == 1 else (1 - be[sn])
                                if bt[s, sn] > 0 and e > 0 and lb[d+1, sn] > -1e300:
                                    vals.append(np.log(bt[s, sn]) + np.log(e) + lb[d+1, sn])
                            if vals:
                                m = max(vals)
                                lb[d, s] = m + np.log(sum(np.exp(v - m) for v in vals))
                    lev = la[D-1, :]
                    mev = np.max(lev)
                    if mev > -1e300:
                        lev_ok = mev + np.log(sum(np.exp(lev - mev)))
                        for d in range(D):
                            lg = la[d] + lb[d] - lev_ok
                            g = np.exp(np.clip(lg, -700, 0))
                            be_num += g * hb[d]
                            be_den += g
                new_be = np.where(be_den > 0, be_num / be_den, be)
                new_be[0] = 0.0
                new_be = np.clip(new_be, 0.01, 0.99)
                new_be[0] = 0.0
                if np.max(np.abs(new_be - be)) < 1e-6:
                    be = new_be
                    break
                be = new_be
            bp0 = _estimate_p0_hmm(bi, bt, be, D)
            bp0 = max(bp0, 0.01)
            boot_N[b] = n / max(1 - bp0, 0.01)
        except Exception:
            boot_N[b] = N_hat

    valid = boot_N[~np.isnan(boot_N)]
    se_boot = float(np.std(valid, ddof=1))
    ci_lower = max(float(np.percentile(valid, 2.5)), n)
    ci_upper = float(np.percentile(valid, 97.5))

    return CRResult(
        N_hat=N_hat,
        variance=se_boot ** 2,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        method='hmm_cr',
        details={
            'n_states': n_states,
            'emit_probs': emit.tolist(),
            'p0': p0,
            'n_observed': n,
            'iterations': iteration + 1,
        }
    )


def _estimate_p0_hmm(init_dist, trans, emit, D):
    """P(never observed across D periods) under HMM."""
    state_dist = init_dist.copy()
    p_never = 1.0
    for d in range(D):
        p_captured_d = np.dot(state_dist, emit)
        p_never *= (1 - p_captured_d)
        state_dist = state_dist @ trans
    return max(p_never, 0.01)


def _compute_variance(Y_arr, x_arr, N_hat):
    """
    Plug-in variance estimator (kept for reference, deprecated in
    favour of bootstrap).
    """
    m = len(Y_arr)
    if N_hat <= 0 or m == 0:
        return 0.0
    q_hat = Y_arr / max(N_hat, 1)
    q_hat = np.clip(q_hat, 0, 1)
    sum_q = q_hat.sum()
    if sum_q <= 0:
        return float('inf')
    numerator = N_hat ** 2 * np.sum(q_hat * (1 - q_hat))
    denominator = sum_q ** 2
    return max(numerator / denominator, 0)


def compute_reserve(N_hat: float, n_observed: int, mean_severity: float,
                    paid_losses: float) -> dict:
    """
    Convert population estimate to actuarial reserve.

    R = N_hat_IBNR * mu_hat - (total expected - paid)
    """
    n_ibnr = max(N_hat - n_observed, 0)
    pure_ibnr_reserve = n_ibnr * mean_severity
    total_ultimate = N_hat * mean_severity
    total_reserve = total_ultimate - paid_losses

    return {
        'N_hat': N_hat,
        'n_observed': n_observed,
        'n_ibnr': n_ibnr,
        'mean_severity': mean_severity,
        'pure_ibnr_reserve': pure_ibnr_reserve,
        'total_ultimate': total_ultimate,
        'paid_losses': paid_losses,
        'total_reserve': total_reserve,
    }
