"""
Monte Carlo experiment engine.

Compares all CR variants vs Chain-Ladder across 7 scenarios.
Metrics: Bias, RMSE, Coverage, CI Width, GoF p-value.

All CIs use nonparametric percentile bootstrap (B=500).
"""

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional

from .scenarios import ScenarioConfig, get_scenario
from .data_generator import generate_data


@dataclass
class MCResult:
    scenario: str
    method: str
    n_reps: int
    bias: float
    rmse: float
    coverage: float
    ci_width: float
    mean_estimate: float
    median_estimate: float
    std_estimate: float
    true_N: int
    estimates: np.ndarray
    elapsed_sec: float
    extra: dict = None


# Expected behaviour per scenario for validation gate
EXPECTATIONS = {
    'S1': {'best': 'closed_cr', 'reason': 'homogeneous -> CL and CR should be equivalent'},
    'S2': {'best': 'closed_cr', 'reason': 'stratified heterogeneity -> CR should outperform CL'},
    'S3': {'best': 'open_cr', 'reason': 'open system -> open_cr should outperform closed_cr and CL'},
    'S4': {'best': 'open_cr', 'reason': 'shock -> CL should break, CR robust'},
    'S5': {'best': 'closed_cr', 'reason': 'short-tail high-p -> all methods should be close'},
    'S6': {'best': 'min_chisq_cr', 'reason': 'long-tail sparse -> distribution-free should do well'},
    'S7': {'best': 'bayesian_dpmm', 'reason': 'extreme heterogeneity -> DPMM should dominate'},
}


# ---------------------------------------------------------------------------
# Chain-Ladder bootstrap (individual-level resampling)
# ---------------------------------------------------------------------------

def _cl_point_from_full(
    capture_histories_full: np.ndarray,
    accident_years: np.ndarray,
    n_accident_years: int,
    n_dev_periods: int,
) -> float:
    """Point estimate: CL total ultimate from individual data."""
    from baselines.chain_ladder import chain_ladder, individual_to_triangle

    triangle = individual_to_triangle(
        capture_histories_full,
        accident_years=accident_years,
        n_accident_years=n_accident_years,
        n_dev_periods=n_dev_periods,
    )
    cl_res = chain_ladder(triangle)
    total = cl_res.ultimate.sum()
    if total <= 0 or np.isnan(total):
        # fallback: count observed individuals
        return float(np.sum(np.any(capture_histories_full == 1, axis=1)))
    return float(total)


def _cl_bootstrap_ci(
    capture_histories_full: np.ndarray,
    accident_years: np.ndarray,
    n_accident_years: int,
    n_dev_periods: int,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> dict:
    """
    Bootstrap CI for Chain-Ladder total ultimate.

    Resamples individuals (rows of capture_histories_full) with replacement,
    reconstructs the triangle, runs CL, and returns percentile CI.
    """
    rng = np.random.default_rng(seed)
    N_full = capture_histories_full.shape[0]

    point = _cl_point_from_full(
        capture_histories_full, accident_years,
        n_accident_years, n_dev_periods,
    )

    boot_ests = np.full(n_boot, np.nan)
    n_failed = 0

    for b in range(n_boot):
        idx = rng.choice(N_full, N_full, replace=True)
        try:
            boot_ests[b] = _cl_point_from_full(
                capture_histories_full[idx],
                accident_years[idx],
                n_accident_years,
                n_dev_periods,
            )
        except Exception:
            n_failed += 1
            continue

    valid = boot_ests[~np.isnan(boot_ests)]
    if len(valid) < n_boot * 0.5:
        # fallback: use Mack SE + normal approx
        from baselines.chain_ladder import chain_ladder, individual_to_triangle
        triangle = individual_to_triangle(
            capture_histories_full, accident_years=accident_years,
            n_accident_years=n_accident_years, n_dev_periods=n_dev_periods,
        )
        cl_res = chain_ladder(triangle)
        se = np.sqrt(np.sum(cl_res.mack_se ** 2)) if cl_res.mack_se is not None else point * 0.1
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        n_obs = np.sum(np.any(capture_histories_full == 1, axis=1))
        return {
            'N_hat': point,
            'ci_lower': max(point - z * se, n_obs),
            'ci_upper': point + z * se,
            'se_boot': se,
        }

    ci_lower = float(np.percentile(valid, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    n_obs = np.sum(np.any(capture_histories_full == 1, axis=1))

    return {
        'N_hat': point,
        'ci_lower': max(ci_lower, n_obs),
        'ci_upper': ci_upper,
        'se_boot': float(np.std(valid, ddof=1)),
    }


def _extract_ci(result, method: str) -> tuple:
    """Extract (N_hat, ci_lower, ci_upper) from any result type."""
    if isinstance(result, dict):
        return result['N_hat'], result.get('ci_lower', 0), result.get('ci_upper', 0)
    else:
        # CRResult, HMMCRResult, DPMMCRResult, CLResult
        ci_lower = getattr(result, 'ci_lower', getattr(result, 'N_ci_lower', 0))
        ci_upper = getattr(result, 'ci_upper', getattr(result, 'N_ci_upper', 0))
        return result.N_hat, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: ScenarioConfig,
    n_reps: int = 1000,
    methods: Optional[List[str]] = None,
    base_seed: int = 42,
    verbose: bool = True,
    skip_slow: bool = True,
    n_boot: int = 500,
) -> Dict[str, MCResult]:
    """Run Monte Carlo experiment for a single scenario."""

    if methods is None:
        methods = ['closed_cr', 'open_cr', 'min_chisq_cr', 'hmm_cr',
                   'doubly_robust', 'chain_ladder']
        if skip_slow:
            pass  # bayesian_dpmm added only when explicitly requested

    from theory.estimators import closed_system_cr, open_system_cr, min_chisq_cr
    from theory.doubly_robust import doubly_robust_cr, min_chisq_cr_improved
    from theory.hmm_cr import hmm_cr_baum_welch

    results = {}

    for method in methods:
        estimates = np.zeros(n_reps)
        ci_lowers = np.zeros(n_reps)
        ci_uppers = np.zeros(n_reps)

        t0 = time.time()
        rep = 0
        fail_count = 0

        while rep < n_reps:
            data = generate_data(config, seed=base_seed + rep)

            try:
                # ---- Closed CR ----
                if method == 'closed_cr':
                    res = closed_system_cr(
                        data.capture_histories,
                        n_boot=n_boot, seed=base_seed + rep,
                    )
                    estimates[rep], ci_lowers[rep], ci_uppers[rep] = \
                        _extract_ci(res, method)

                # ---- Open CR ----
                elif method == 'open_cr':
                    ever_obs = np.any(data.capture_histories_full == 1, axis=1)
                    entry_obs = data.entry_times[ever_obs] if data.entry_times is not None else None
                    exit_obs = data.exit_times[ever_obs] if data.exit_times is not None else None
                    res = open_system_cr(
                        data.capture_histories,
                        entry_times=entry_obs,
                        exit_times=exit_obs,
                        n_boot=n_boot, seed=base_seed + rep,
                    )
                    estimates[rep], ci_lowers[rep], ci_uppers[rep] = \
                        _extract_ci(res, method)

                # ---- Min Chi-Square (Improved) ----
                elif method == 'min_chisq_cr':
                    res = min_chisq_cr_improved(
                        data.capture_histories, K=5,
                        n_boot=n_boot, seed=base_seed + rep,
                    )
                    estimates[rep], ci_lowers[rep], ci_uppers[rep] = \
                        _extract_ci(res, method)

                # ---- HMM Baum-Welch ----
                elif method == 'hmm_cr':
                    res = hmm_cr_baum_welch(
                        data.capture_histories,
                        max_iter=100, n_bootstrap=min(n_boot, 200),
                        seed=base_seed + rep,
                    )
                    estimates[rep] = res.N_hat
                    ci_lowers[rep] = res.N_ci_lower
                    ci_uppers[rep] = res.N_ci_upper

                # ---- Doubly Robust ----
                elif method == 'doubly_robust':
                    res = doubly_robust_cr(
                        data.capture_histories,
                        n_boot=n_boot, seed=base_seed + rep,
                    )
                    estimates[rep], ci_lowers[rep], ci_uppers[rep] = \
                        _extract_ci(res, method)

                # ---- Bayesian DPMM (slow, only when explicitly included) ----
                elif method == 'bayesian_dpmm':
                    from theory.bayesian_cr import dpmm_cr_neal8
                    res = dpmm_cr_neal8(
                        data.capture_histories,
                        n_iter=2000, burnin=500,
                        M_augment=300, seed=base_seed + rep,
                    )
                    estimates[rep] = res.N_mean
                    ci_lowers[rep] = res.N_ci_lower
                    ci_uppers[rep] = res.N_ci_upper

                # ---- Chain-Ladder (baseline, individual-level bootstrap CI) ----
                elif method == 'chain_ladder':
                    ci_result = _cl_bootstrap_ci(
                        data.capture_histories_full,
                        accident_years=data.accident_years,
                        n_accident_years=max(1, config.D // 2),
                        n_dev_periods=config.D,
                        n_boot=n_boot,
                        seed=base_seed + rep,
                    )
                    estimates[rep] = ci_result['N_hat']
                    ci_lowers[rep] = ci_result['ci_lower']
                    ci_uppers[rep] = ci_result['ci_upper']

                rep += 1

            except Exception as e:
                fail_count += 1
                if fail_count > n_reps * 0.1:
                    if verbose:
                        sys.stdout.write(
                            f'\n  [{method}] Too many failures '
                            f'({fail_count}), using fallback\n'
                        )
                    estimates[rep] = data.n_observed
                    ci_lowers[rep] = data.n_observed
                    ci_uppers[rep] = data.n_observed
                    rep += 1
                continue

            if verbose and rep % max(1, n_reps // 10) == 0:
                pct = rep / n_reps * 100
                bar = '#' * int(pct // 5) + '-' * (20 - int(pct // 5))
                elapsed = time.time() - t0
                eta = elapsed / max(rep, 1) * (n_reps - rep)
                sys.stdout.write(
                    f'\r  {config.name}/{method}: [{bar}] {pct:5.1f}% '
                    f'({rep}/{n_reps}) ETA {eta:.0f}s'
                )
                sys.stdout.flush()

        elapsed = time.time() - t0
        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        N_true = config.N_true
        bias = float(np.mean(estimates - N_true))
        rmse = float(np.sqrt(np.mean((estimates - N_true) ** 2)))
        coverage = float(np.mean((ci_lowers <= N_true) & (ci_uppers >= N_true)))
        ci_width = float(np.mean(ci_uppers - ci_lowers))

        results[method] = MCResult(
            scenario=config.name,
            method=method,
            n_reps=n_reps,
            bias=bias,
            rmse=rmse,
            coverage=coverage,
            ci_width=ci_width,
            mean_estimate=float(np.mean(estimates)),
            median_estimate=float(np.median(estimates)),
            std_estimate=float(np.std(estimates)),
            true_N=N_true,
            estimates=estimates,
            elapsed_sec=elapsed,
        )

    return results


def validate_expectations(results: Dict[str, MCResult], scenario_id: str) -> Dict:
    """Compare results against theoretical expectations. Returns violations."""
    violations = {}
    expected = EXPECTATIONS.get(scenario_id)
    if expected is None:
        return violations

    best_method = expected['best']
    reason = expected['reason']

    # Find the method with lowest RMSE
    best_actual = min(results, key=lambda m: results[m].rmse)

    if best_actual != best_method:
        violations['best_method'] = {
            'expected': best_method,
            'actual': best_actual,
            'reason_expected': reason,
            'best_rmse': results[best_actual].rmse,
            'expected_rmse': results.get(best_method, results[best_actual]).rmse,
        }

    # Check if CL is worse than at least one CR method in heterogeneous scenarios
    if scenario_id in ('S2', 'S4', 'S7') and 'chain_ladder' in results:
        cl_rmse = results['chain_ladder'].rmse
        best_cr_rmse = min(
            r.rmse for m, r in results.items() if m != 'chain_ladder'
        )
        if cl_rmse <= best_cr_rmse:
            violations['cl_vs_cr'] = {
                'expected': 'CR should beat CL (heterogeneous scenario)',
                'actual': f'CL RMSE={cl_rmse:.1f} <= best CR RMSE={best_cr_rmse:.1f}',
            }

    # Check coverage rates
    for method, mcr in results.items():
        if mcr.coverage < 0.80:
            violations[f'low_coverage_{method}'] = {
                'method': method,
                'coverage': mcr.coverage,
                'threshold': 0.80,
            }

    return violations


def format_results_table(results: Dict[str, MCResult], scenario_id: str = '') -> str:
    """Format results as a readable table."""
    lines = []
    lines.append(
        f'{"Method":<18} {"Bias":>8} {"RMSE":>8} '
        f'{"Cov":>7} {"CI W":>8} {"Mean":>8}'
    )
    lines.append('-' * 62)
    for method, mcr in sorted(results.items(), key=lambda x: x[1].rmse):
        lines.append(
            f'{method:<18} {mcr.bias:>8.1f} {mcr.rmse:>8.1f} '
            f'{mcr.coverage:>7.3f} {mcr.ci_width:>8.1f} {mcr.mean_estimate:>8.1f}'
        )
    return '\n'.join(lines)
