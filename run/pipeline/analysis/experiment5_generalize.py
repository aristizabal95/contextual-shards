"""Experiment 5: Apply shard detection pipeline to a new environment.

Usage (after training a MiniGrid agent):
    uv run python run/pipeline/analysis/experiment5_generalize.py
"""


def main() -> None:
    print("Experiment 5: Generalization Pipeline")
    print("=" * 50)
    print("Same pipeline as Experiments 1+2, different environment.")
    print()
    print("Steps:")
    print("  1. Train agent on MiniGrid or custom 2D env with competing rewards")
    print("  2. Collect activations: collect_activations.py environment=minigrid")
    print("  3. Run experiment1_probing.py on new activation dataset")
    print("  4. Run experiment1_causal.py")
    print("  5. Run experiment2_separability.py")
    print("  6. Compare shard candidates across environments (qualitative)")
    print()
    print("Candidate MiniGrid envs:")
    print("  - MiniGrid-FourRooms-v0  (navigation + multiple exits)")
    print("  - Custom: two reward sources in 2D grid (cheese + corner analogue)")


if __name__ == "__main__":
    main()
