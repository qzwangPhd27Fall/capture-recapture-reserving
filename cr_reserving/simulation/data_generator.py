"""
Data generator for CR reserving simulation experiments.

Generates individual-level claim data with configurable:
  - Population size (including IBNR)
  - Capture probabilities (homogeneous/heterogeneous)
  - Open/closed system dynamics
  - Severity distributions
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .scenarios import ScenarioConfig


@dataclass
class SimulatedData:
    capture_histories: np.ndarray   # (n_observed, D) - observed claims only
    capture_histories_full: np.ndarray  # (N_true, D) - all claims including IBNR
    N_true: int
    n_observed: int
    n_ibnr: int
    severities: np.ndarray          # (N_true,) - ultimate severity per claim
    group_labels: Optional[np.ndarray]  # (N_true,) - group/cluster assignment
    entry_times: Optional[np.ndarray]   # (N_true,) - entry period
    exit_times: Optional[np.ndarray]    # (N_true,) - exit period
    accident_years: Optional[np.ndarray]  # (N_true,) - accident year
    scenario: str


def generate_data(config: ScenarioConfig, seed: Optional[int] = None) -> SimulatedData:
    """
    Generate simulated claim data according to scenario configuration.

    Parameters
    ----------
    config : ScenarioConfig
        Scenario specification.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SimulatedData with capture histories, severities, and metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    N = config.N_true
    D = config.D

    # Assign groups
    if config.heterogeneity in ('stratified', 'dpmm') and config.group_weights is not None:
        group_labels = np.random.choice(config.n_groups, size=N, p=config.group_weights)
    else:
        group_labels = np.zeros(N, dtype=int)

    # Assign entry/exit times for open system
    if config.is_open:
        entry_times = np.zeros(N, dtype=int)
        n_initial = int(N * 0.7)
        n_entering = N - n_initial
        entry_times[:n_initial] = 0
        if n_entering > 0:
            entry_times[n_initial:] = np.random.randint(1, D, n_entering)

        exit_times = np.full(N, D + 1, dtype=int)
        for j in range(N):
            for d in range(entry_times[j] + 1, D):
                if np.random.random() < config.exit_rate:
                    exit_times[j] = d
                    break
    else:
        entry_times = np.zeros(N, dtype=int)
        exit_times = np.full(N, D + 1, dtype=int)

    # Generate capture histories
    capture_histories = np.zeros((N, D), dtype=int)

    for j in range(N):
        g = group_labels[j]

        if config.heterogeneity in ('stratified', 'dpmm') and config.group_probs is not None:
            probs = config.group_probs[g]
        else:
            probs = config.capture_probs

        for d in range(D):
            if d < entry_times[j] or d >= exit_times[j]:
                continue
            if np.random.random() < probs[min(d, len(probs) - 1)]:
                capture_histories[j, d] = 1

    # Identify observed vs IBNR
    ever_observed = np.any(capture_histories == 1, axis=1)
    n_observed = ever_observed.sum()
    n_ibnr = N - n_observed

    observed_indices = np.where(ever_observed)[0]
    observed_histories = capture_histories[observed_indices]

    # Generate severities
    severities = np.abs(np.random.normal(config.severity_mean, config.severity_sd, N))

    # Assign accident years (uniform across available years)
    n_ay = max(1, D // 2)
    accident_years = np.random.randint(0, n_ay, N)

    return SimulatedData(
        capture_histories=observed_histories,
        capture_histories_full=capture_histories,
        N_true=N,
        n_observed=n_observed,
        n_ibnr=n_ibnr,
        severities=severities,
        group_labels=group_labels,
        entry_times=entry_times,
        exit_times=exit_times,
        accident_years=accident_years,
        scenario=config.name,
    )


def generate_batch(config: ScenarioConfig, n_reps: int = 1000,
                   base_seed: int = 42) -> list:
    """Generate multiple replications for Monte Carlo experiments."""
    datasets = []
    for rep in range(n_reps):
        data = generate_data(config, seed=base_seed + rep)
        datasets.append(data)
    return datasets
