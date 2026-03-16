"""Experiment 3: Reinforcement distribution correlation analysis (D4).

Tests D4: corner shard strength correlates with corner reinforcement frequency.

Usage (after training agents and running Experiment 1 on each):
    uv run python run/pipeline/analysis/experiment3_dist.py
"""
from src.analysis_module.statistics.stat_tests import report_experiment3_correlation


def main() -> None:
    print("Experiment 3: D4 Reinforcement Traceability")
    print("=" * 50)
    print("Prerequisites:")
    print("  - 3 agents trained (corner_biased, uniform, anti_corner)")
    print("  - Experiment 1 pipeline run on each agent's activations")
    print()
    print("Steps:")
    print("  1. Load corner shard strength from each agent's Exp 1 causal results")
    print("  2. Call report_experiment3_correlation()")
    print("  3. Check: Spearman rho > 0.7 (target per research plan)")
    print()
    print("Example (with placeholder values):")
    result = report_experiment3_correlation(
        agent_names=["corner_biased", "uniform", "anti_corner"],
        corner_reinforcement_freqs=[0.75, 0.25, 0.10],
        corner_shard_strengths=[0.80, 0.40, 0.12],
    )
    print(f"  rho={result['spearman_rho']:.3f}, p={result['p_value']:.4f}, "
          f"meets_target={result['meets_target']}")


if __name__ == "__main__":
    main()
