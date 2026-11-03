"""
Unified nonparametric bootstrap CI for all CR estimators.

Replaces buggy plugin variance estimators with a single, well-tested
percentile bootstrap that works uniformly across methods.
"""

import numpy as np
from numpy.random import default_rng
from typing import Optional, Callable, Tuple
from scipy.stats import norm as norm_dist


def bootstrap_ci(
    point_estimator: Callable[..., float],
    *args,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Nonparametric percentile bootstrap CI.

    Resamples rows of the first argument (capture_histories) with
    replacement, re-estimates N_hat each time, and returns
    percentile-based confidence limits.

    Parameters
    ----------
    point_estimator : callable
        Function that takes (*args) and returns a float point estimate.
        The first arg should be the (n_obs, D) capture_histories matrix.
    *args : arrays
        Arguments to point_estimator. First arg is resampled along axis 0.
        Additional args are also resampled consistently if they have the
        same first dimension.
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (0.05 → 95% CI).
    seed : int, optional
        Random seed.
    lower_bound : float
        Floor for the lower CI bound (e.g. n_observed).
    verbose : bool
        Print progress bar.

    Returns
    -------
    dict with keys: N_hat (original point estimate), ci_lower, ci_upper,
    se_boot, boot_samples, n_failed.
    """
    rng = default_rng(seed)

    # Point estimate on original data
    N_hat = float(point_estimator(*args))
    n_rows = args[0].shape[0]

    boot_N = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        idx = rng.choice(n_rows, n_rows, replace=True)
        # Resample all args consistently along axis 0
        boot_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.shape[0] == n_rows:
                boot_args.append(arg[idx])
            else:
                boot_args.append(arg)

        try:
            boot_N[b] = float(point_estimator(*boot_args))
        except Exception:
            n_failed += 1
            continue

        if verbose and b % max(1, n_boot // 10) == 0:
            import sys
            pct = (b + 1) / n_boot * 100
            sys.stdout.write(f'\r  bootstrap: {pct:.0f}% ({b+1}/{n_boot})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    valid = boot_N[~np.isnan(boot_N)]
    if len(valid) < n_boot * 0.5:
        raise RuntimeError(
            f'Bootstrap failed: only {len(valid)}/{n_boot} valid replicates'
        )

    ci_lower = float(np.percentile(valid, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    se_boot = float(np.std(valid, ddof=1))

    return {
        'N_hat': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_boot': se_boot,
        'boot_samples': valid,
        'n_failed': n_failed,
    }


def parametric_bootstrap_cr(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Parametric bootstrap CI for capture-recapture population size.

    Unlike the nonparametric bootstrap (which only resamples observed
    individuals and misses IBNR variability), this method:

    1. Fits the Mt model: p̂_d = per-period capture probabilities
       from the observed data (column means).
    2. Estimates N̂ from the observed data.
    3. Parametric bootstrap: generates N̂ synthetic individuals
       with capture probs p̂_d, keeps only those with >=1 capture,
       re-applies the point estimator to get N̂*.
    4. Returns percentile CI from the N̂* distribution.

    This accounts for BOTH sampling variability in the observed
    individuals AND variability in the IBNR layer.

    Parameters
    ----------
    point_estimator : callable(capture_histories, *extra_args) → float
    capture_histories : (n_obs, D) binary matrix
    *extra_args : passed through to point_estimator
    n_boot, alpha, seed, lower_bound, verbose : standard
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape

    # Step 1: Point estimate on original data
    N_hat = float(point_estimator(capture_histories, *extra_args))

    # Pre-compute sufficient statistics for Beta posterior
    n_d = capture_histories.sum(axis=0).astype(float)

    # Step 2: Parametric bootstrap with perturbed p_hat (Beta posterior)
    # Per-iteration Beta sampling propagates p_hat estimation uncertainty.
    boot_N = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        # Sample p_d from Beta posterior (Jeffreys prior)
        p_star = rng.beta(n_d + 0.5, np.maximum(N_hat - n_d, 1.0) + 0.5)
        p_star = np.clip(p_star, 0.005, 0.995)

        N_gen = max(int(np.ceil(N_hat)), n_obs + 1)
        sim_full = rng.random((N_gen, D)) < p_star[np.newaxis, :]
        ever_obs = sim_full.any(axis=1)
        sim_obs = sim_full[ever_obs]

        if len(sim_obs) == 0:
            n_failed += 1
            continue

        try:
            boot_N[b] = float(point_estimator(sim_obs, *extra_args))
        except Exception:
            n_failed += 1
            continue

        if verbose and b % max(1, n_boot // 10) == 0:
            import sys
            pct = (b + 1) / n_boot * 100
            sys.stdout.write(f'\r  par-bootstrap: {pct:.0f}% ({b+1}/{n_boot})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    valid = boot_N[~np.isnan(boot_N)]
    if len(valid) < n_boot * 0.5:
        raise RuntimeError(
            f'Parametric bootstrap failed: only {len(valid)}/{n_boot} valid replicates'
        )

    # Step 4: BC (bias-corrected) percentile adjustment
    # z0 = Phi^{-1}(proportion of bootstrap estimates below original estimate)
    prop_below = np.mean(valid < N_hat)
    # Clamp away from 0/1 to avoid infinite z-scores
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = float(norm_dist.ppf(prop_below))
    z_alpha = norm_dist.ppf(alpha / 2)

    # BC percentile adjustment: α₁, α₂
    alpha1 = float(norm_dist.cdf(2 * z0 + z_alpha))
    alpha2 = float(norm_dist.cdf(2 * z0 - z_alpha))  # = norm_dist.cdf(2*z0 + z_{1-α/2})

    # Overdispersion inflation
    f_i = capture_histories.sum(axis=1)
    p_mean = float(np.mean(f_i)) / D
    var_binom = D * p_mean * (1.0 - p_mean)
    var_obs = float(np.var(f_i, ddof=1))
    c_hat = max(1.0, var_obs / max(var_binom, 0.001))

    if c_hat > 1.05:
        median_boot = float(np.median(valid))
        inflated = median_boot + np.sqrt(c_hat) * (valid - median_boot)
        ci_lower = float(np.percentile(inflated, 100 * alpha1))
        ci_upper = float(np.percentile(inflated, 100 * alpha2))
        se_boot = float(np.std(valid, ddof=1)) * np.sqrt(c_hat)
    else:
        ci_lower = float(np.percentile(valid, 100 * alpha1))
        ci_upper = float(np.percentile(valid, 100 * alpha2))
        se_boot = float(np.std(valid, ddof=1))

    return {
        'N_hat': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_boot': se_boot,
        'boot_samples': valid,
        'n_failed': n_failed,
        'overdispersion_c': float(c_hat),
    }


def parametric_bootstrap_bc(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Bias-corrected parametric bootstrap CI.

    Standard parametric bootstrap CIs are centred on the (potentially
    biased) point estimate.  This version estimates the finite-sample
    bias from the bootstrap distribution and subtracts it, giving a
    bias-corrected point estimate and CI.

    Steps:
      1. N̂ = point_estimate(data)
      2. Parametric bootstrap from N̂ → N̂* (B samples)
      3. bias_est = mean(N̂*) - N̂
      4. N̂_bc = N̂ - bias_est = 2·N̂ - mean(N̂*)
      5. N̂*_bc = N̂* - bias_est  (bias-corrected replicates)
      6. CI = percentiles of N̂*_bc

    Parameters
    ----------
    point_estimator : callable(capture_histories, *extra_args) → float
    capture_histories : (n_obs, D) binary matrix
    *extra_args : passed through to point_estimator
    n_boot, alpha, seed, lower_bound, verbose : standard
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape

    N_hat = float(point_estimator(capture_histories, *extra_args))

    # Pre-compute sufficient statistics for Beta posterior
    n_d = capture_histories.sum(axis=0).astype(float)

    boot_N = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        # Sample p_d from Beta posterior (Jeffreys prior)
        p_star = rng.beta(n_d + 0.5, np.maximum(N_hat - n_d, 1.0) + 0.5)
        p_star = np.clip(p_star, 0.005, 0.995)

        N_gen = max(int(np.ceil(N_hat)), n_obs + 1)
        sim_full = rng.random((N_gen, D)) < p_star[np.newaxis, :]
        ever_obs = sim_full.any(axis=1)
        sim_obs = sim_full[ever_obs]

        if len(sim_obs) == 0:
            n_failed += 1
            continue

        try:
            boot_N[b] = float(point_estimator(sim_obs, *extra_args))
        except Exception:
            n_failed += 1
            continue

    valid = boot_N[~np.isnan(boot_N)]
    if len(valid) < n_boot * 0.5:
        raise RuntimeError(
            f'Bias-corrected bootstrap failed: only {len(valid)}/{n_boot} valid'
        )

    # Bias correction
    bias_est = float(np.mean(valid)) - N_hat
    N_bc = N_hat - bias_est  # = 2*N_hat - mean(valid)
    boot_bc = valid - bias_est

    # Overdispersion inflation for heterogeneous data
    # (Explicit bias correction already handles bias; BC percentile not needed)
    f_i = capture_histories.sum(axis=1)
    p_mean = float(np.mean(f_i)) / D
    var_binom = D * p_mean * (1.0 - p_mean)
    var_obs = float(np.var(f_i, ddof=1))
    c_hat = max(1.0, var_obs / max(var_binom, 0.001))

    if c_hat > 1.05:
        median_boot_bc = float(np.median(boot_bc))
        inflated = median_boot_bc + np.sqrt(c_hat) * (boot_bc - median_boot_bc)
        ci_lower = float(np.percentile(inflated, 100 * alpha / 2))
        ci_upper = float(np.percentile(inflated, 100 * (1 - alpha / 2)))
        se_boot = float(np.std(valid, ddof=1)) * np.sqrt(c_hat)
    else:
        ci_lower = float(np.percentile(boot_bc, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_bc, 100 * (1 - alpha / 2)))
        se_boot = float(np.std(boot_bc, ddof=1))

    return {
        'N_hat': max(N_bc, lower_bound),
        'N_hat_raw': N_hat,
        'bias_est': bias_est,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_boot': se_boot,
        'boot_samples': valid,
        'boot_bc_samples': boot_bc,
        'n_failed': n_failed,
        'overdispersion_c': float(c_hat),
    }


def bootstrap_ci_multiarrays(
    point_estimator: Callable[..., float],
    arrays: dict,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
) -> dict:
    """
    Bootstrap CI when estimator takes multiple named arrays.

    All arrays in `arrays` are resampled consistently (same row indices).
    The estimator receives **arrays (original or resampled).

    Parameters
    ----------
    point_estimator : callable(**arrays) → float
    arrays : dict of str → np.ndarray
        All must have the same first dimension.
    n_boot, alpha, seed, lower_bound : as in bootstrap_ci
    """
    rng = default_rng(seed)

    N_hat = float(point_estimator(**arrays))

    # Find common first dimension
    n_rows = None
    for arr in arrays.values():
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError(
                    f'All arrays must have same first dimension, '
                    f'got {n_rows} and {arr.shape[0]}'
                )

    if n_rows is None:
        raise ValueError('No array arguments found for bootstrap resampling')

    boot_N = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        idx = rng.choice(n_rows, n_rows, replace=True)
        boot_arrays = {}
        for key, arr in arrays.items():
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_rows:
                boot_arrays[key] = arr[idx]
            else:
                boot_arrays[key] = arr

        try:
            boot_N[b] = float(point_estimator(**boot_arrays))
        except Exception:
            n_failed += 1
            continue

    valid = boot_N[~np.isnan(boot_N)]
    if len(valid) < n_boot * 0.5:
        raise RuntimeError(
            f'Bootstrap failed: only {len(valid)}/{n_boot} valid replicates'
        )

    return {
        'N_hat': N_hat,
        'ci_lower': max(float(np.percentile(valid, 100 * alpha / 2)), lower_bound),
        'ci_upper': float(np.percentile(valid, 100 * (1 - alpha / 2))),
        'se_boot': float(np.std(valid, ddof=1)),
        'boot_samples': valid,
        'n_failed': n_failed,
    }


def jackknife_sandwich_ci(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    alpha: float = 0.05,
    lower_bound: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Sandwich variance CI via delete-1 jackknife influence function.

    For M-estimators N̂ that solve Σ ψ(X_i; N̂) = 0, the jackknife
    pseudo-values approximate the influence function:

        IF_i ≈ (n-1) · (N̂ - N̂_{(-i)})

    The sandwich variance is:

        Var(N̂) = (1 / n(n-1)) · Σ_i (θ̃_i - θ̄)²

    where θ̃_i = n·N̂ - (n-1)·N̂_{(-i)} are the pseudo-values.

    This is consistent under weak conditions and does not rely on
    parametric bootstrap assumptions about the DGP.

    Parameters
    ----------
    point_estimator : callable(capture_histories, *extra_args) → float
    capture_histories : (n_obs, D) binary matrix
    *extra_args : passed through to point_estimator
    alpha : significance level
    lower_bound : floor for lower CI bound
    verbose : show progress

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, se_sandwich, n_jackknife
    """
    from scipy.stats import norm as norm_dist

    n_obs = capture_histories.shape[0]
    N_hat = float(point_estimator(capture_histories, *extra_args))

    # Delete-1 jackknife
    N_drop = np.full(n_obs, np.nan)
    n_ok = 0

    for i in range(n_obs):
        # Delete row i
        keep = np.ones(n_obs, dtype=bool)
        keep[i] = False
        X_drop = capture_histories[keep]

        # Also drop corresponding rows from extra_args
        extra_drop = []
        for arg in extra_args:
            if isinstance(arg, np.ndarray) and arg.shape[0] == n_obs:
                extra_drop.append(arg[keep])
            else:
                extra_drop.append(arg)

        try:
            N_drop[i] = float(point_estimator(X_drop, *extra_drop))
            n_ok += 1
        except Exception:
            continue

        if verbose and i % max(1, n_obs // 10) == 0:
            import sys
            pct = (i + 1) / n_obs * 100
            sys.stdout.write(f'\r  jackknife: {pct:.0f}% ({i+1}/{n_obs})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    if n_ok < n_obs * 0.5:
        raise RuntimeError(
            f'Jackknife failed: only {n_ok}/{n_obs} valid drop-one estimates'
        )

    # Pseudo-values
    pseudo = n_obs * N_hat - (n_obs - 1) * N_drop
    pseudo_mean = float(np.mean(pseudo))
    se_sandwich = float(np.std(pseudo, ddof=1) / np.sqrt(n_obs))

    # Normal-based CI with sandwich SE
    z = norm_dist.ppf(1 - alpha / 2)
    ci_lower = pseudo_mean - z * se_sandwich
    ci_upper = pseudo_mean + z * se_sandwich

    return {
        'N_hat': max(pseudo_mean, lower_bound),
        'N_hat_raw': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_sandwich': se_sandwich,
        'n_jackknife': n_ok,
    }


def asymptotic_sandwich_ci(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    alpha: float = 0.05,
    lower_bound: float = 0.0,
    estimator_type: str = 'doubly_robust',
    calibrate: bool = True,
    n_calib: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Analytical sandwich variance CI with bootstrap calibration.

    The raw delta-method SE uses plug-in estimates (N̂, q̂₀) in the
    variance formula, which can substantially underestimate the true
    sampling variability when q̂₀ differs from q₀ due to sampling error.

    When calibrate=True, a fast parametric bootstrap (B=n_calib) with
    Poisson-randomised N_gen estimates the inflation factor needed.
    This corrects for the plug-in variance underestimation.

    Parameters
    ----------
    point_estimator : callable → float
    capture_histories : (n_obs, D) binary matrix
    estimator_type : 'doubly_robust' or 'min_chisq'
    alpha : significance level
    lower_bound : floor for lower CI bound
    calibrate : bootstrap-calibrate the SE (default True)
    n_calib : bootstrap replicates for calibration (default 100)
    seed : random seed
    verbose : show progress bar

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, se_asymptotic, calibration_factor
    """
    from scipy.stats import norm as norm_dist

    n_obs, D = capture_histories.shape
    n_d = capture_histories.sum(axis=0).astype(float)
    N_hat = float(point_estimator(capture_histories, *extra_args))

    if estimator_type == 'doubly_robust':
        se_raw = _doubly_robust_delta_se(n_obs, n_d, N_hat, D)
    elif estimator_type == 'min_chisq':
        se_raw = _min_chisq_sandwich_se(capture_histories, N_hat, D, *extra_args)
    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")

    # Bootstrap calibration of SE inflation
    if calibrate:
        rng = default_rng(seed)
        p_d = np.clip(n_d / n_obs, 0.01, 0.99)

        boot_N = np.full(n_calib, np.nan)
        n_failed = 0

        for b in range(n_calib):
            # Poisson-randomised N_gen: propagates population-size uncertainty
            N_gen = max(int(rng.poisson(N_hat)), n_obs + 1)
            # Beta-perturbed p_star: propagates detection-prob uncertainty
            p_star = rng.beta(n_d + 0.5, np.maximum(N_hat - n_d, 1.0) + 0.5)
            p_star = np.clip(p_star, 0.005, 0.995)

            sim_full = rng.random((N_gen, D)) < p_star[np.newaxis, :]
            sim_obs = sim_full[sim_full.any(axis=1)]

            if len(sim_obs) == 0:
                n_failed += 1
                continue

            try:
                boot_N[b] = float(point_estimator(sim_obs, *extra_args))
            except Exception:
                n_failed += 1
                continue

            if verbose and b % max(1, n_calib // 10) == 0:
                import sys
                pct = (b + 1) / n_calib * 100
                sys.stdout.write(f'\r  calibrate: {pct:.0f}% ({b+1}/{n_calib})')
                sys.stdout.flush()

        if verbose:
            import sys
            sys.stdout.write('\n')
            sys.stdout.flush()

        valid = boot_N[~np.isnan(boot_N)]
        if len(valid) >= 30:
            se_boot = float(np.std(valid, ddof=1))
            # Calibration factor: bootstrap SE / asymptotic SE
            inflation = max(1.0, se_boot / max(se_raw, 0.01))
            # Cap to avoid extreme inflation from small n_calib
            inflation = min(inflation, 3.0)
            # Bias correction: N̂_bc = 2·N̂ - mean(N̂*)
            # If estimator is downward-biased, mean(N̂*) < N̂ and N̂_bc > N̂
            bias_est = float(np.mean(valid)) - N_hat
            N_bc = N_hat - bias_est
        else:
            inflation = 1.0
            N_bc = N_hat

        se = se_raw * inflation
    else:
        se = se_raw
        inflation = 1.0
        N_bc = N_hat

    z = norm_dist.ppf(1 - alpha / 2)
    ci_lower = N_bc - z * se
    ci_upper = N_bc + z * se

    return {
        'N_hat': max(N_bc, lower_bound),
        'N_hat_raw': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_asymptotic': se,
        'se_raw': se_raw,
        'calibration_factor': float(inflation),
    }


def _doubly_robust_delta_se(
    n_obs: float, n_d: np.ndarray, N_hat: float, D: int
) -> float:
    """
    Delta-method SE for doubly-robust estimator.

    N_dr = n / (1 - Π(1 - n_d/n))

    Gradient w.r.t. (n, n_1, ..., n_D) derived analytically.
    Covariance from independent-Bernoulli capture model.
    """
    q_d = 1.0 - n_d / n_obs  # 1 - p_d per period
    q0 = float(np.prod(q_d))  # Π(1-p_d) = prob(never captured)
    denom = 1.0 - q0

    # Gradient of N_dr
    # ∂N/∂n = 1/(1-q0) + (n·q0/(1-q0)²) · Σ_d [n_d/(n(n-n_d))]
    dN_dn = 1.0 / denom
    for d in range(D):
        if n_obs > n_d[d] > 0:
            dN_dn += (n_obs * q0 / denom**2) * (n_d[d] / (n_obs * (n_obs - n_d[d])))

    # ∂N/∂n_d = -n·q0 / [(1-q0)² · (n - n_d)]
    dN_dnd = np.zeros(D)
    for d in range(D):
        if n_obs > n_d[d]:
            dN_dnd[d] = -n_obs * q0 / (denom**2 * (n_obs - n_d[d]))

    # Covariance matrix of (n, n_1, ..., n_D)
    # Under independent-Bernoulli capture model with p_d = Prob(capture in d):
    #   Var(n) = N·(1-q0)·q0, Var(n_d) = N·p_d·(1-p_d)
    #   Cov(n, n_d) = N·p_d·q0, Cov(n_d, n_k) = 0 (d≠k)
    p_d = 1.0 - q_d
    var_n = N_hat * denom * q0
    var_nd = N_hat * p_d * (1.0 - p_d)
    cov_n_nd = N_hat * p_d * q0

    # Sandwich: Var = ∇F^T · Σ · ∇F
    var_total = dN_dn**2 * var_n
    for d in range(D):
        var_total += dN_dnd[d]**2 * var_nd[d]
        var_total += 2 * dN_dn * dN_dnd[d] * cov_n_nd[d]

    return float(np.sqrt(max(var_total, 1e-6)))


def _min_chisq_sandwich_se(
    capture_histories: np.ndarray, N_hat: float, D: int, *extra_args
) -> float:
    """
    Sandwich SE for min-chisq estimator via estimating equations.

    The estimator solves Ψ(N) = ∂χ²/∂N = 0 where:
        χ²(N) = Σ_{k=1}^D (O_k - N·b_k)² / (N·b_k)

    Sandwich: Var(N̂) = B / A² where
        A = E[-∂Ψ/∂N] (sensitivity matrix, scalar here)
        B = E[Ψ²]    (variability of estimating equation)

    We estimate Ψ_i (contribution of individual i) via the influence
    function implied by the capture-frequency distribution.
    """
    from math import comb

    n_obs = capture_histories.shape[0]
    f_i = capture_histories.sum(axis=1)
    p_hat = np.clip(float(np.mean(f_i)) / D, 0.01, 0.99)

    # Binomial probabilities under fitted model
    b = np.zeros(D + 1)
    for k in range(D + 1):
        b[k] = comb(D, k) * (p_hat ** k) * ((1 - p_hat) ** (D - k))

    # Observed capture-frequency counts
    O = np.array([np.sum(f_i == k) for k in range(1, D + 1)], dtype=float)

    # ---- Sensitivity A = E[-∂²χ²/∂N²] / 2 (evaluated at N̂) ----
    # ∂χ²/∂N = -Σ_k (O_k² - N²b_k²) / (N² b_k)
    # ∂²χ²/∂N² = Σ_k 2·O_k² / (N³ b_k)
    A = 0.0
    for k_idx, k in enumerate(range(1, D + 1)):
        if b[k] > 1e-10 and N_hat > 0:
            A += 2.0 * O[k_idx]**2 / (N_hat**3 * b[k])
    A = max(A, 1e-10)

    # ---- Meat B = Var(Ψ) via influence function ----
    # Ψ_i ≈ contribution of individual i to the estimating equation
    # For capture count k_i, the contribution is:
    #   ψ_i = -(I(k_i=k) - N·b_k) / (N·b_k) summed over k with weights
    # Simplified: use the observed Fisher information as a proxy
    # B = (1/n) Σ_i ψ_i² ≈ Σ_k (O_k/n) · [∂χ²_k/∂N]²

    B = 0.0
    for k_idx, k in enumerate(range(1, D + 1)):
        if b[k] > 1e-10 and n_obs > 0:
            # Score contribution for an individual with k captures
            score_k = -(O[k_idx] - N_hat * b[k]) / (N_hat * b[k] + 0.5)
            weight = O[k_idx] / n_obs
            B += weight * score_k**2

    B = max(B, 1e-10)

    # Sandwich variance
    var_sandwich = B / (A**2)
    return float(np.sqrt(max(var_sandwich, 1e-6)))


def smoothed_frequency_bootstrap_ci(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
    overdispersion: float = 1.0,
    verbose: bool = False,
) -> dict:
    """
    Smoothed bootstrap CI via sufficient-statistic resampling.

    Unlike the parametric bootstrap which fixes N_gen = ceil(N_hat),
    this method bootstraps the sufficient statistics (n, n_d, O_k)
    directly, with an overdispersed Poisson draw for the total
    observed count n* to propagate population-size uncertainty.

    The smoothed frequency distribution is sampled from a Dirichlet
    posterior, avoiding the rigid binomial assumption.

    Parameters
    ----------
    point_estimator : callable(capture_histories, *extra_args) → float
    capture_histories : (n_obs, D) binary matrix
    *extra_args : passed through to point_estimator
    n_boot, alpha, seed, lower_bound, verbose : standard
    overdispersion : multiplier on Poisson variance for n*
                     (1.0 = Poisson, >1 = overdispersed)

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, se_boot, n_failed
    """
    rng = default_rng(seed)
    n_obs, D = capture_histories.shape

    N_hat = float(point_estimator(capture_histories, *extra_args))

    # Sufficient statistics
    n_d = capture_histories.sum(axis=0).astype(float)
    f_i = capture_histories.sum(axis=1).astype(int)
    O_k = np.array([np.sum(f_i == k) for k in range(1, D + 1)], dtype=float)

    # Dirichlet smoothing prior
    alpha_prior = 0.5
    alpha_post_freq = O_k + alpha_prior
    alpha_post_col = n_d + alpha_prior

    # Per-period weights for reconstructing which periods
    col_w = np.clip(n_d / n_obs, 0.01, 0.99)
    col_w = col_w / col_w.sum()

    boot_N = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        # 1. Draw total observed count n* with overdispersion
        # Poisson-Gamma mixture: overdisp_factor > 1 inflates variance
        if overdispersion > 1.0:
            # Gamma-Poisson = NegBin
            scale = overdispersion - 1.0
            n_star = rng.poisson(rng.gamma(n_obs / scale, scale))
        else:
            n_star = rng.poisson(n_obs)
        n_star = max(n_star, D + 1)

        # 2. Sample capture frequency distribution from smoothed Dirichlet
        freq_probs = rng.dirichlet(alpha_post_freq)
        f_star = rng.choice(np.arange(1, D + 1), size=n_star, p=freq_probs)

        # 3. Reconstruct capture histories
        sim_obs = np.zeros((n_star, D), dtype=bool)
        for i in range(n_star):
            k = int(f_star[i])
            if k >= D:
                sim_obs[i, :] = True
            elif k > 0:
                cols = rng.choice(D, size=k, replace=False, p=col_w)
                sim_obs[i, cols] = True

        # 4. Estimate N* from reconstructed data
        try:
            boot_N[b] = float(point_estimator(sim_obs, *extra_args))
        except Exception:
            n_failed += 1
            continue

        if verbose and b % max(1, n_boot // 10) == 0:
            import sys
            pct = (b + 1) / n_boot * 100
            sys.stdout.write(f'\r  smoothed boot: {pct:.0f}% ({b+1}/{n_boot})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    valid = boot_N[~np.isnan(boot_N)]
    if len(valid) < n_boot * 0.5:
        raise RuntimeError(
            f'Smoothed bootstrap failed: only {len(valid)}/{n_boot} valid'
        )

    ci_lower = float(np.percentile(valid, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    se_boot = float(np.std(valid, ddof=1))

    return {
        'N_hat': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_boot': se_boot,
        'boot_samples': valid,
        'n_failed': n_failed,
        'n_star_mean': float(n_obs),
    }


def bootstrap_calibrated_ci(
    point_estimator: Callable[..., float],
    capture_histories: np.ndarray,
    *extra_args,
    n_boot_outer: int = 500,
    n_boot_inner: int = 100,
    n_calib: int = 40,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    lower_bound: float = 0.0,
    lambda_grid: Optional[list] = None,
    verbose: bool = False,
) -> dict:
    """
    Bootstrap-calibrated CI via double (nested) parametric bootstrap.

    Standard percentile CI from a parametric bootstrap may under-cover when
    the estimator has finite-sample bias or the bootstrap SE underestimates
    the true sampling variability.  This function uses Efron's double
    bootstrap calibration:

      1. Run B_outer parametric bootstrap → outer distribution Ĝ
      2. For M pilot outer samples, run B_inner inner bootstrap → inner Ĝ_b
      3. For each λ in a grid, estimate actual coverage:
           ĉ(λ) = (1/M) Σ_b 1{N̂ ∈ [Ĝ_b^{-1}(λ/2), Ĝ_b^{-1}(1−λ/2)]}
      4. Find λ̂ such that ĉ(λ̂) ≈ 1−α
      5. Return CI at level λ̂: [Ĝ^{-1}(λ̂/2), Ĝ^{-1}(1−λ̂/2)]

    The calibrated λ is returned so it can be reused across MC replicates
    from the same DGP (avoiding per-rep double-bootstrap cost).

    Parameters
    ----------
    point_estimator : callable(capture_histories, *extra_args) → float
    capture_histories : (n_obs, D) binary matrix
    *extra_args : passed through to point_estimator
    n_boot_outer : outer bootstrap replicates (default 500)
    n_boot_inner : inner bootstrap replicates per pilot (default 100)
    n_calib : number of pilot samples for calibration (default 40)
    alpha : desired significance level (default 0.05 → 95% CI)
    seed : random seed
    lower_bound : floor for lower CI bound
    lambda_grid : candidate λ values (default: [0.02, 0.05, 0.10, 0.20, 0.35, 0.50])
    verbose : show progress

    Returns
    -------
    dict with N_hat, ci_lower, ci_upper, se_boot, lambda_calibrated,
    outer_samples, n_failed
    """
    if lambda_grid is None:
        lambda_grid = [0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]

    rng = default_rng(seed)
    n_obs, D = capture_histories.shape
    n_d = capture_histories.sum(axis=0).astype(float)

    # ---- Outer bootstrap ----
    N_hat = float(point_estimator(capture_histories, *extra_args))

    outer_boot = np.full(n_boot_outer, np.nan)
    outer_data = []  # store the first M bootstrap datasets for inner loop
    n_failed_outer = 0

    for b in range(n_boot_outer):
        p_star = rng.beta(n_d + 0.5, np.maximum(N_hat - n_d, 1.0) + 0.5)
        p_star = np.clip(p_star, 0.005, 0.995)
        N_gen = max(int(np.ceil(N_hat)), n_obs + 1)
        sim_full = rng.random((N_gen, D)) < p_star[np.newaxis, :]
        sim_obs = sim_full[sim_full.any(axis=1)]

        if len(sim_obs) == 0:
            n_failed_outer += 1
            if b < n_calib:
                outer_data.append(None)
            continue

        try:
            outer_boot[b] = float(point_estimator(sim_obs, *extra_args))
        except Exception:
            n_failed_outer += 1
            if b < n_calib:
                outer_data.append(None)
            continue

        # Store first M valid outer datasets for inner calibration
        if b < n_calib:
            outer_data.append(sim_obs)

        if verbose and b % max(1, n_boot_outer // 10) == 0:
            import sys
            pct = (b + 1) / n_boot_outer * 100
            sys.stdout.write(f'\r  outer boot: {pct:.0f}% ({b+1}/{n_boot_outer})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    valid_outer = outer_boot[~np.isnan(outer_boot)]
    if len(valid_outer) < n_boot_outer * 0.5:
        raise RuntimeError(
            f'Outer bootstrap failed: only {len(valid_outer)}/{n_boot_outer} valid'
        )

    # ---- Inner bootstrap (calibration) ----
    actual_cov = {lam: 0.0 for lam in lambda_grid}
    n_calib_valid = 0

    for m, sim_obs_m in enumerate(outer_data):
        if sim_obs_m is None or len(sim_obs_m) < 2:
            continue

        n_m, D_m = sim_obs_m.shape
        n_d_m = sim_obs_m.sum(axis=0).astype(float)
        N_hat_m = float(point_estimator(sim_obs_m, *extra_args))

        inner_boot = np.full(n_boot_inner, np.nan)
        for ib in range(n_boot_inner):
            p_star_inner = rng.beta(n_d_m + 0.5, n_m - n_d_m + 0.5)
            p_star_inner = np.clip(p_star_inner, 0.005, 0.995)
            N_gen_inner = max(int(np.ceil(N_hat_m)), n_m + 1)
            sim_full_inner = rng.random((N_gen_inner, D_m)) < p_star_inner[np.newaxis, :]
            sim_obs_inner = sim_full_inner[sim_full_inner.any(axis=1)]

            if len(sim_obs_inner) == 0:
                continue

            try:
                inner_boot[ib] = float(point_estimator(sim_obs_inner, *extra_args))
            except Exception:
                continue

        valid_inner = inner_boot[~np.isnan(inner_boot)]
        if len(valid_inner) < n_boot_inner * 0.3:
            continue

        n_calib_valid += 1

        # For each λ, check if N_hat is covered by the inner CI
        for lam in lambda_grid:
            lo = float(np.percentile(valid_inner, 100 * lam / 2))
            hi = float(np.percentile(valid_inner, 100 * (1 - lam / 2)))
            if lo <= N_hat <= hi:
                actual_cov[lam] += 1

        if verbose and m % max(1, n_calib // 5) == 0:
            import sys
            pct = (m + 1) / n_calib * 100
            sys.stdout.write(f'\r  inner calib: {pct:.0f}% ({m+1}/{n_calib})')
            sys.stdout.flush()

    if verbose:
        import sys
        sys.stdout.write('\n')
        sys.stdout.flush()

    if n_calib_valid < 5:
        # Fallback: use raw outer bootstrap percentiles
        lo = float(np.percentile(valid_outer, 100 * alpha / 2))
        hi = float(np.percentile(valid_outer, 100 * (1 - alpha / 2)))
        return {
            'N_hat': N_hat,
            'ci_lower': max(lo, lower_bound),
            'ci_upper': hi,
            'se_boot': float(np.std(valid_outer, ddof=1)),
            'lambda_calibrated': alpha,
            'n_calib_valid': n_calib_valid,
            'n_failed': n_failed_outer,
        }

    # Normalise coverage counts
    for lam in lambda_grid:
        actual_cov[lam] /= n_calib_valid

    # Find λ̂: the λ whose actual coverage is closest to target 1-α
    target = 1.0 - alpha
    best_lam = alpha
    best_dist = float('inf')
    for lam in lambda_grid:
        dist = abs(actual_cov[lam] - target)
        if dist < best_dist:
            best_dist = dist
            best_lam = lam

    # Linear interpolation between bracketing λ values for better accuracy
    sorted_lams = sorted(lambda_grid)
    cov_vals = [actual_cov[lam] for lam in sorted_lams]
    lam_cal = best_lam
    for i in range(len(sorted_lams) - 1):
        c1, c2 = cov_vals[i], cov_vals[i + 1]
        l1, l2 = sorted_lams[i], sorted_lams[i + 1]
        if (c1 - target) * (c2 - target) <= 0 and abs(c2 - c1) > 1e-10:
            # Target lies between these two points — linear interpolation
            t_frac = (target - c1) / (c2 - c1)
            lam_cal = l1 + t_frac * (l2 - l1)
            break

    # Final CI at calibrated λ
    ci_lower = float(np.percentile(valid_outer, 100 * lam_cal / 2))
    ci_upper = float(np.percentile(valid_outer, 100 * (1 - lam_cal / 2)))

    return {
        'N_hat': N_hat,
        'ci_lower': max(ci_lower, lower_bound),
        'ci_upper': ci_upper,
        'se_boot': float(np.std(valid_outer, ddof=1)),
        'lambda_calibrated': float(lam_cal),
        'lambda_grid': lambda_grid,
        'actual_coverage': {lam: actual_cov[lam] for lam in lambda_grid},
        'n_calib_valid': n_calib_valid,
        'n_failed': n_failed_outer,
    }
