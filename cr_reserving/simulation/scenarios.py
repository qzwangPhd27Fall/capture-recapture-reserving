"""
Scenario configurations for Monte Carlo simulation experiments.

S1: Closed system, homogeneous capture probability
S2: Closed system, heterogeneous capture probability (by line of business)
S3: Open system, stable entry/exit
S4: Open system, shock (e.g., COVID-like disruption)
S5: Short-tail line (e.g., auto physical damage)
S6: Long-tail line (e.g., liability)
S7: High-dimensional heterogeneity (DPMM vs stratified)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ScenarioConfig:
    name: str
    description: str
    N_true: int
    D: int
    capture_probs: np.ndarray
    is_open: bool = False
    entry_rate: float = 0.0
    exit_rate: float = 0.0
    heterogeneity: str = 'none'
    n_groups: int = 1
    group_probs: Optional[List[np.ndarray]] = None
    group_weights: Optional[np.ndarray] = None
    shock_period: Optional[int] = None
    shock_multiplier: float = 1.0
    severity_mean: float = 10000.0
    severity_sd: float = 5000.0

    def __post_init__(self):
        if isinstance(self.capture_probs, list):
            self.capture_probs = np.array(self.capture_probs)


def get_scenario(scenario_id: str) -> ScenarioConfig:
    """Get a pre-defined scenario configuration."""
    scenarios = {
        'S1': _scenario_s1(),
        'S2': _scenario_s2(),
        'S3': _scenario_s3(),
        'S4': _scenario_s4(),
        'S5': _scenario_s5(),
        'S6': _scenario_s6(),
        'S7': _scenario_s7(),
    }
    if scenario_id not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(scenarios.keys())}")
    return scenarios[scenario_id]


def get_all_scenarios() -> dict:
    """Return all 7 scenarios."""
    return {sid: get_scenario(sid) for sid in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']}


def _scenario_s1() -> ScenarioConfig:
    """S1: Closed system, homogeneous capture probability."""
    D = 10
    p = 0.15
    return ScenarioConfig(
        name='S1',
        description='Closed system, homogeneous p=0.15',
        N_true=500,
        D=D,
        capture_probs=np.full(D, p),
        is_open=False,
        heterogeneity='none',
    )


def _scenario_s2() -> ScenarioConfig:
    """S2: Closed system, heterogeneous capture probability (3 LOBs)."""
    D = 10
    return ScenarioConfig(
        name='S2',
        description='Closed system, 3 groups with different p',
        N_true=500,
        D=D,
        capture_probs=np.full(D, 0.15),
        is_open=False,
        heterogeneity='stratified',
        n_groups=3,
        group_probs=[
            np.full(D, 0.25),  # fast-reporting (auto)
            np.full(D, 0.12),  # medium (property)
            np.full(D, 0.05),  # slow (liability)
        ],
        group_weights=np.array([0.5, 0.3, 0.2]),
    )


def _scenario_s3() -> ScenarioConfig:
    """S3: Open system, stable entry/exit rates."""
    D = 10
    return ScenarioConfig(
        name='S3',
        description='Open system, 5% entry/exit per period',
        N_true=500,
        D=D,
        capture_probs=np.full(D, 0.15),
        is_open=True,
        entry_rate=0.05,
        exit_rate=0.05,
        heterogeneity='none',
    )


def _scenario_s4() -> ScenarioConfig:
    """S4: Open system, shock at period 5 (COVID-like)."""
    D = 10
    probs = np.full(D, 0.15)
    probs[4:7] = 0.05  # reporting drops during shock
    return ScenarioConfig(
        name='S4',
        description='Open system, shock at period 5 (p drops to 0.05)',
        N_true=500,
        D=D,
        capture_probs=probs,
        is_open=True,
        entry_rate=0.03,
        exit_rate=0.03,
        heterogeneity='none',
        shock_period=5,
        shock_multiplier=0.33,
    )


def _scenario_s5() -> ScenarioConfig:
    """S5: Short-tail (auto PD), high initial p, fast decay."""
    D = 6
    probs = np.array([0.40, 0.25, 0.10, 0.05, 0.02, 0.01])
    return ScenarioConfig(
        name='S5',
        description='Short-tail (6 periods), high initial p',
        N_true=500,
        D=D,
        capture_probs=probs,
        is_open=False,
        heterogeneity='none',
        severity_mean=5000,
        severity_sd=2000,
    )


def _scenario_s6() -> ScenarioConfig:
    """S6: Long-tail (liability), slow build-up of capture probability."""
    D = 15
    probs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.13,
                       0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02])
    return ScenarioConfig(
        name='S6',
        description='Long-tail (15 periods), slow detection',
        N_true=500,
        D=D,
        capture_probs=probs,
        is_open=False,
        heterogeneity='none',
        severity_mean=50000,
        severity_sd=30000,
    )


def _scenario_s7() -> ScenarioConfig:
    """S7: High-dimensional heterogeneity (5 latent clusters)."""
    D = 10
    return ScenarioConfig(
        name='S7',
        description='5 latent clusters, extreme heterogeneity',
        N_true=500,
        D=D,
        capture_probs=np.full(D, 0.15),
        is_open=False,
        heterogeneity='dpmm',
        n_groups=5,
        group_probs=[
            np.full(D, 0.35),  # very fast
            np.full(D, 0.20),  # fast
            np.full(D, 0.10),  # medium
            np.full(D, 0.04),  # slow
            np.full(D, 0.01),  # very slow (near-invisible)
        ],
        group_weights=np.array([0.15, 0.25, 0.30, 0.20, 0.10]),
    )
