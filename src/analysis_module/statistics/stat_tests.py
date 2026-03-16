"""Statistical tests for shard theory experiments."""
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.01,
) -> List[bool]:
    """Return True for each p-value that survives Bonferroni correction.

    Args:
        p_values: list of p-values (e.g., one per layer comparison)
        alpha: family-wise error rate (default 0.01 per research plan)

    Returns:
        list of booleans — True = null hypothesis rejected after correction
    """
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [bool(p < threshold) for p in p_values]


def spearman_correlation(
    x: List[float],
    y: List[float],
) -> Tuple[float, float]:
    """Compute Spearman rank correlation.

    Args:
        x: first variable (e.g., reinforcement frequencies)
        y: second variable (e.g., shard strengths)

    Returns:
        (rho, p_value) — Spearman rho and two-tailed p-value
    """
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group1, group2: 1D arrays of measurements

    Returns:
        Cohen's d (signed: positive if group1 > group2)
    """
    diff = float(group1.mean() - group2.mean())
    pooled_std = float(
        np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2) + 1e-8
    )
    return diff / pooled_std


def report_experiment3_correlation(
    agent_names: List[str],
    corner_reinforcement_freqs: List[float],
    corner_shard_strengths: List[float],
) -> Dict:
    """Test D4: Spearman correlation between reinforcement distribution and shard strength.

    Args:
        agent_names: e.g. ["corner_biased", "uniform", "anti_corner"]
        corner_reinforcement_freqs: training corner frequency per agent
        corner_shard_strengths: measured corner shard strength per agent

    Returns:
        Dict with spearman_rho, p_value, significant, meets_target (rho > 0.7)
    """
    rho, p = spearman_correlation(corner_reinforcement_freqs, corner_shard_strengths)
    return {
        "agents": agent_names,
        "corner_reinforcement_freqs": corner_reinforcement_freqs,
        "corner_shard_strengths": corner_shard_strengths,
        "spearman_rho": rho,
        "p_value": p,
        "significant": bool(p < 0.05),
        "meets_target": bool(rho > 0.7),
    }
