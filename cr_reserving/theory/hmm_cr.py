"""
HMM-based capture-recapture estimator for insurance claims reserving.

Models the claim lifecycle as a hidden Markov process with states:
  U = Unreported (IBNR)
  R = Reported (not yet paying)
  A = Active (paying/adjusting)
  S = Settled
  Re = Reopened

The HMM framework provides:
  - Full Baum-Welch EM estimation of transition and emission matrices
  - State-path inference via Viterbi decoding
  - Population size estimation from P(never observed | theta)
  - Bootstrap confidence intervals
"""

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


# State labels
U, R, A, S, Re = 0, 1, 2, 3, 4
N_STATES = 5
STATE_NAMES = ['U', 'R', 'A', 'S', 'Re']


@dataclass
class HMMCRResult:
    N_hat: float
    N_ci_lower: float
    N_ci_upper: float
    p0: float
    trans_matrix: np.ndarray
    emit_matrix: np.ndarray
    init_dist: np.ndarray
    state_paths: Optional[np.ndarray] = None
    log_likelihood: float = 0.0
    n_iter: int = 0
    converged: bool = False
    details: dict = None

    def actuarial_params(self) -> Dict[str, np.ndarray]:
        """Extract actuarially meaningful parameters from transition matrix."""
        T = self.trans_matrix
        return {
            'reporting_rate': T[U, R],        # P(report next | unreported)
            'payment_start_rate': T[R, A],     # P(start paying | reported)
            'settlement_rate': T[A, S] + T[R, S],  # P(settle | active/reported)
            'reopen_rate': T[S, Re],           # P(reopen | settled)
            'reclose_rate': T[Re, S] + T[Re, R],  # P(close again | reopened)
        }


def hmm_cr_baum_welch(
    capture_histories: np.ndarray,
    n_states: int = N_STATES,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    structural_constraints: bool = True,
    seed: Optional[int] = None,
) -> HMMCRResult:
    """
    Full Baum-Welch EM for HMM-based capture-recapture.

    Parameters
    ----------
    capture_histories : (n_obs, D) binary matrix
    n_states : number of hidden states (default 5: U,R,A,S,Re)
    max_iter : maximum EM iterations
    tol : convergence tolerance on log-likelihood
    n_bootstrap : bootstrap replicates for CI
    alpha : significance level
    structural_constraints : enforce insurance-lifecycle constraints on T
    seed : random seed

    Returns
    -------
    HMMCRResult with N_hat, trans/emit matrices, state paths
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape

    # ---- Initialization ----
    trans, emit, init = _initialize_hmm(n_states, structural_constraints, rng)

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # ---- E-step: forward-backward ----
        log_alpha, log_beta, log_lik = _forward_backward(
            capture_histories, trans, emit, init
        )

        # ---- Check convergence ----
        if abs(log_lik - prev_ll) < tol:
            break
        prev_ll = log_lik

        # ---- Compute posteriors (gamma, xi) ----
        gamma, xi = _compute_posteriors(
            capture_histories, log_alpha, log_beta, log_lik, trans, emit
        )

        # ---- M-step: update parameters ----
        trans, emit, init = _m_step(
            gamma, xi, capture_histories, structural_constraints
        )

    converged = (iteration < max_iter - 1)

    # ---- Population size estimation ----
    p0 = _compute_p0(trans, emit, init, D)
    N_hat = n_obs / max(1.0 - p0, 0.01)

    # ---- Bootstrap CI ----
    boot_N = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n_obs, n_obs, replace=True)
        boot_hist = capture_histories[idx]
        try:
            bt, be, bi = _initialize_hmm(n_states, structural_constraints, rng)
            for _ in range(50):  # fewer iters for bootstrap
                la, lb, ll = _forward_backward(boot_hist, bt, be, bi)
                bg, bx = _compute_posteriors(boot_hist, la, lb, ll, bt, be)
                bt, be, bi = _m_step(bg, bx, boot_hist, structural_constraints)
            bp0 = _compute_p0(bt, be, bi, D)
            boot_N[b] = n_obs / max(1.0 - bp0, 0.01)
        except Exception:
            boot_N[b] = N_hat

    N_ci_lower = max(np.percentile(boot_N, 2.5), n_obs)
    N_ci_upper = np.percentile(boot_N, 97.5)

    # ---- Viterbi decoding for most likely state paths ----
    state_paths = _viterbi_decode(capture_histories, trans, emit, init)

    # ---- Actuarial parameter extraction ----
    details = {}
    if n_states == N_STATES:
        details['actuarial_params'] = {
            'reporting_rate': float(trans[U, R]),
            'settlement_rate': float(trans[A, S]),
            'reopen_rate': float(trans[S, Re]),
        }

    return HMMCRResult(
        N_hat=N_hat,
        N_ci_lower=N_ci_lower,
        N_ci_upper=N_ci_upper,
        p0=p0,
        trans_matrix=trans,
        emit_matrix=emit,
        init_dist=init,
        state_paths=state_paths,
        log_likelihood=log_lik,
        n_iter=iteration + 1,
        converged=converged,
        details=details,
    )


# ---- Core HMM algorithms ----

def _initialize_hmm(
    n_states: int, structural: bool, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize HMM parameters with sensible defaults."""

    # Transition matrix
    trans = np.zeros((n_states, n_states))
    if n_states == N_STATES and structural:
        # Enforce structural zeros (lifecycle constraints)
        trans[U, U] = 0.6; trans[U, R] = 0.4
        trans[R, R] = 0.2; trans[R, A] = 0.6; trans[R, S] = 0.2
        trans[A, A] = 0.4; trans[A, S] = 0.5; trans[A, R] = 0.1
        trans[S, S] = 0.85; trans[S, Re] = 0.15
        trans[Re, Re] = 0.2; trans[Re, S] = 0.5; trans[Re, R] = 0.3
    else:
        # Random initialization with row normalization
        for i in range(n_states):
            row = rng.dirichlet(np.ones(n_states))
            trans[i] = row

    # Emission matrix: P(observed | state) — probability of a "capture" at any period
    # For insurance: observation = payment activity or status change
    emit = np.zeros(n_states)
    if n_states == N_STATES:
        emit[U] = 0.0    # Unreported → never observed
        emit[R] = 0.3    # Reported → sometimes activity
        emit[A] = 0.8    # Active → usually activity
        emit[S] = 0.05   # Settled → rarely activity
        emit[Re] = 0.7   # Reopened → usually activity
    else:
        emit = rng.uniform(0.1, 0.9, n_states)
        emit[0] = 0.0  # first state = unreported/unobserved

    emit = np.clip(emit, 0.01, 0.99)
    emit[0] = 0.01  # small prob for numerical stability

    # Initial distribution
    init = np.zeros(n_states)
    if n_states == N_STATES:
        init[U] = 0.4; init[R] = 0.3; init[A] = 0.2; init[S] = 0.08; init[Re] = 0.02
    else:
        init = rng.dirichlet(np.ones(n_states))
    init /= init.sum()

    return trans, emit, init


def _forward_backward(
    X: np.ndarray, trans: np.ndarray, emit: np.ndarray, init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward algorithm for HMM."""
    n_obs, D = X.shape
    n_states = len(emit)

    # log-space for numerical stability
    log_trans = np.log(np.maximum(trans, 1e-300))
    log_init = np.log(np.maximum(init, 1e-300))

    log_alpha = np.full((n_obs, D, n_states), -np.inf)
    log_beta = np.full((n_obs, D, n_states), -np.inf)

    total_ll = 0.0

    for j in range(n_obs):
        # Forward
        for s in range(n_states):
            e = emit[s] if X[j, 0] == 1 else (1.0 - emit[s])
            if init[s] > 0 and e > 0:
                log_alpha[j, 0, s] = log_init[s] + np.log(e)

        for d in range(1, D):
            for s in range(n_states):
                e = emit[s] if X[j, d] == 1 else (1.0 - emit[s])
                if e <= 0:
                    continue
                vals = []
                for sp in range(n_states):
                    if trans[sp, s] > 0 and log_alpha[j, d - 1, sp] > -1e300:
                        vals.append(log_alpha[j, d - 1, sp] + log_trans[sp, s])
                if vals:
                    log_alpha[j, d, s] = logsumexp(vals) + np.log(e)

        # Sequence log-likelihood
        ll_vals = log_alpha[j, D - 1, :]
        max_ll = np.max(ll_vals)
        if max_ll > -1e300:
            total_ll += max_ll + np.log(np.sum(np.exp(ll_vals - max_ll)))

        # Backward
        log_beta[j, D - 1, :] = 0.0
        for d in range(D - 2, -1, -1):
            for s in range(n_states):
                vals = []
                for sn in range(n_states):
                    e = emit[sn] if X[j, d + 1] == 1 else (1.0 - emit[sn])
                    if trans[s, sn] > 0 and e > 0 and log_beta[j, d + 1, sn] > -1e300:
                        vals.append(log_trans[s, sn] + np.log(e) + log_beta[j, d + 1, sn])
                if vals:
                    log_beta[j, d, s] = logsumexp(vals)

    return log_alpha, log_beta, total_ll


def _compute_posteriors(
    X: np.ndarray,
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_lik: float,
    trans: np.ndarray,
    emit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute state and transition posteriors (gamma, xi)."""
    n_obs, D, n_states = log_alpha.shape

    gamma = np.zeros((n_obs, D, n_states))
    xi = np.zeros((n_states, n_states, D - 1))

    for j in range(n_obs):
        # gamma[j,d,s] = P(S_d = s | O_j)
        for d in range(D):
            log_gamma = log_alpha[j, d] + log_beta[j, d]
            log_gamma -= np.max(log_gamma)
            gamma[j, d] = np.exp(log_gamma)
            gamma[j, d] /= gamma[j, d].sum()

        # xi[s, s', d] for sequence j
        for d in range(D - 1):
            for s in range(n_states):
                for sn in range(n_states):
                    e = emit[sn] if X[j, d + 1] == 1 else (1.0 - emit[sn])
                    if trans[s, sn] <= 0 or e <= 0:
                        continue
                    log_xi = (log_alpha[j, d, s] + np.log(trans[s, sn]) +
                              np.log(e) + log_beta[j, d + 1, sn])
                    if log_xi > -1e300:
                        xi[s, sn, d] += np.exp(log_xi)

    return gamma, xi


def _m_step(
    gamma: np.ndarray,
    xi: np.ndarray,
    X: np.ndarray,
    structural: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update HMM parameters from expected sufficient statistics."""
    n_obs, D, n_states = gamma.shape

    # Update transition matrix
    trans_new = np.zeros((n_states, n_states))

    for s in range(n_states):
        row_sum = 0.0
        for sn in range(n_states):
            trans_new[s, sn] = xi[s, sn].sum()
            row_sum += trans_new[s, sn]

        if row_sum > 0:
            trans_new[s] /= row_sum
        else:
            trans_new[s, s] = 1.0  # absorbing

    if structural and n_states == N_STATES:
        # Enforce structural zeros
        trans_new[U, A] = 0; trans_new[U, S] = 0; trans_new[U, Re] = 0
        trans_new[S, U] = 0; trans_new[S, R] = 0; trans_new[S, A] = 0
        trans_new = np.where(trans_new < 0, 0, trans_new)
        # re-normalize
        for s in range(n_states):
            rs = trans_new[s].sum()
            if rs > 0:
                trans_new[s] /= rs
            else:
                trans_new[s, s] = 1.0

    # Update emission probabilities
    emit_new = np.zeros(n_states)
    for s in range(n_states):
        num = 0.0
        den = 0.0
        for j in range(n_obs):
            for d in range(D):
                num += gamma[j, d, s] * X[j, d]
                den += gamma[j, d, s]
        emit_new[s] = num / max(den, 1e-10)
    emit_new = np.clip(emit_new, 0.01, 0.99)
    emit_new[U] = 0.01  # unreported state: minimal emission

    # Update initial distribution
    init_new = gamma[:, 0, :].sum(axis=0)
    init_new /= init_new.sum()

    return trans_new, emit_new, init_new


def _compute_p0(
    trans: np.ndarray, emit: np.ndarray, init: np.ndarray, D: int
) -> float:
    """Compute P(never observed across D periods)."""
    n_states = len(emit)

    # State distribution over time starting from init
    state_dist = init.copy()
    p_never = 1.0

    for d in range(D):
        # probability of being observed at period d
        p_obs_d = np.dot(state_dist, emit)
        p_never *= (1.0 - p_obs_d)
        # transition to next period
        state_dist = state_dist @ trans

    return max(p_never, 0.001)


def _viterbi_decode(
    X: np.ndarray, trans: np.ndarray, emit: np.ndarray, init: np.ndarray
) -> np.ndarray:
    """Viterbi decoding: most likely state path for each sequence."""
    n_obs, D = X.shape
    n_states = len(emit)

    log_trans = np.log(np.maximum(trans, 1e-300))
    log_init = np.log(np.maximum(init, 1e-300))

    paths = np.zeros((n_obs, D), dtype=int)

    for j in range(n_obs):
        delta = np.full((D, n_states), -np.inf)
        psi = np.zeros((D, n_states), dtype=int)

        for s in range(n_states):
            e = emit[s] if X[j, 0] == 1 else (1.0 - emit[s])
            if e > 0:
                delta[0, s] = log_init[s] + np.log(e)

        for d in range(1, D):
            for s in range(n_states):
                e = emit[s] if X[j, d] == 1 else (1.0 - emit[s])
                if e <= 0:
                    continue
                best_val = -np.inf
                best_sp = 0
                for sp in range(n_states):
                    val = delta[d - 1, sp] + log_trans[sp, s]
                    if val > best_val:
                        best_val = val
                        best_sp = sp
                if best_val > -1e300:
                    delta[d, s] = best_val + np.log(e)
                    psi[d, s] = best_sp

        # backtrack
        paths[j, D - 1] = int(np.argmax(delta[D - 1]))
        for d in range(D - 2, -1, -1):
            paths[j, d] = psi[d + 1, paths[j, d + 1]]

    return paths


def logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    if len(x) == 0:
        return -np.inf
    m = np.max(x)
    if m == -np.inf:
        return -np.inf
    return float(m + np.log(np.sum(np.exp(x - m))))
