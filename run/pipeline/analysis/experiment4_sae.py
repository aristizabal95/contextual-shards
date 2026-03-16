"""Experiment 4: SAE feature discovery and comparison with supervised probes.

Tests whether an unsupervised SAE recovers features corresponding to shards
identified by supervised probing (cross-method convergence).

Usage (after train_sae.py and experiment1_probing.py):
    uv run python run/pipeline/analysis/experiment4_sae.py
"""


def main() -> None:
    print("Experiment 4: SAE Feature Discovery")
    print("=" * 50)
    print("Steps:")
    print("  1. Load trained SAE from outputs/checkpoints/sae.pt")
    print("  2. Load activations from data/processed/activations.h5")
    print("  3. Compute context profiles: FeatureAnalyzer.compute_context_profiles()")
    print("  4. Identify top features per concept")
    print("  5. Correlate SAE features with probe predictions")
    print("  6. Report: features with Pearson r > 0.7 (target per research plan)")


if __name__ == "__main__":
    main()
