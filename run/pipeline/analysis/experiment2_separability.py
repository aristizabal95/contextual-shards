"""Experiment 2: Shard separability test (D3).

Tests whether shards are causally independent: suppressing cheese shard
should not substantially affect corner behavior, and vice versa.

Usage (after Experiment 1 identifies shard layer):
    uv run python run/pipeline/analysis/experiment2_separability.py
"""


def main() -> None:
    print("Experiment 2: Shard Separability (D3)")
    print("=" * 50)
    print("Prerequisites:")
    print("  - Experiment 1 complete (shard layer identified in outputs/tables/experiment1_candidates.json)")
    print("  - Agent loaded with procgen (Python 3.10 environment)")
    print()
    print("Steps:")
    print("  1. Load shard layer from Experiment 1 candidates")
    print("  2. Collect context-specific activation batches:")
    print("     - cheese_context: cheese within 5 steps")
    print("     - cheese_baseline: cheese absent")
    print("     - corner_context: agent near top-right corner")
    print("     - corner_baseline: agent far from corner")
    print("  3. Run SeparabilityTester.run()")
    print("  4. Report I_cheese_corner and I_corner_cheese (target > 0.8)")
    print()
    print("from src.shard_module.separability.separability_tester import SeparabilityTester")
    print("tester = SeparabilityTester(agent=agent, shard_layer=shard_layer)")
    print("results = tester.run(...)")


if __name__ == "__main__":
    main()
