"""Experiment 4: SAE feature discovery and comparison with supervised probes.

Tests cross-method convergence: do unsupervised SAE features align with
the shard directions found by supervised linear probing?

Target: SAE features with Pearson |r| > 0.7 with probe predictions.

Usage (after train_sae.py and experiment1_probing.py):
    uv run python run/pipeline/analysis/experiment4_sae.py

Prerequisites:
    - outputs/checkpoints/sae.pt (from train_sae.py)
    - data/processed/activations_impala.h5 (from collect_activations.py)
    - outputs/tables/experiment1_probing.json (from experiment1_probing.py)
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def run_sae_analysis(
    sae_path: str = "outputs/checkpoints/sae.pt",
    activations_path: str = "data/processed/activations_impala.h5",
    probing_path: str = "outputs/tables/experiment1_probing.json",
    layer: str = "embedder.fc",
    expansion_factor: int = 8,
    l1_coef: float = 0.01,
    r_threshold: float = 0.7,
    top_k: int = 10,
    output_dir: str = "outputs/tables",
) -> dict:
    import torch
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
    from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
    from src.sae_module.feature.feature_analyzer import FeatureAnalyzer
    from src.probe_module.probe.linear_probe import LinearProbe

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # probing_path is accepted for CLI consistency but probes are re-trained from
    # activations to ensure the correlation is measured on held-out data.
    _ = probing_path

    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}\nRun train_sae.py first.")
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations not found: {activations_path}")

    # Load activations and labels
    logger.info(f"Loading activations from {activations_path}")
    known_concepts = ["cheese_presence", "cheese_proximity", "cheese_direction", "corner_proximity"]
    with HDF5ActivationDataset(activations_path, mode="r") as ds:
        acts = ds.get_all_activations(layer)
        labels = {}
        for k in known_concepts:
            try:
                labels[k] = ds.get_all_labels(k)
            except KeyError:
                pass
    acts_flat = acts.reshape(len(acts), -1)
    d_input = acts_flat.shape[1]

    # Load SAE
    logger.info(f"Loading SAE from {sae_path}")
    sae = SparseAutoencoder(d_input=d_input, expansion_factor=expansion_factor, l1_coef=l1_coef)
    state = torch.load(sae_path, map_location="cpu")
    sae.load_state_dict(state)
    sae.eval()

    analyzer = FeatureAnalyzer(sae)

    # Compute context profiles (contrast vectors)
    logger.info("Computing SAE feature context profiles")
    profiles = analyzer.compute_context_profiles(acts_flat, labels)
    top_features = analyzer.top_features_per_concept(profiles, top_k=top_k)

    # Train probes on a held-out split so SAE-probe correlation is measured out-of-sample.
    # In-sample predictions inflate correlation because any feature correlated with the true
    # label will spuriously match an over-fit probe.
    logger.info("Training probes (80/20 split) to get held-out predictions for correlation")
    rng = np.random.default_rng(42)
    n_total = len(acts_flat)
    perm = rng.permutation(n_total)
    train_n = int(0.8 * n_total)
    train_idx, test_idx = perm[:train_n], perm[train_n:]
    acts_test = acts_flat[test_idx]

    probe_predictions: dict = {}
    for concept, concept_labels in labels.items():
        y = (concept_labels > 0.5).astype(float)
        if y.mean() < 0.01 or y.mean() > 0.99:
            continue
        if len(np.unique(y[train_idx])) < 2:
            logger.warning(f"Skipping probe for '{concept}': single class in train split")
            continue
        probe = LinearProbe()
        probe.fit(acts_flat[train_idx], y[train_idx])
        probe_predictions[concept] = probe.predict(acts_test)

    # Find SAE features that correlate with held-out probe predictions
    logger.info(f"Finding SAE features with |r| >= {r_threshold} on held-out test set")
    matching = analyzer.find_matching_features(acts_test, probe_predictions, r_threshold=r_threshold)

    results = {
        "layer": layer,
        "n_samples": len(acts_flat),
        "d_input": d_input,
        "d_hidden": d_input * expansion_factor,
        "r_threshold": r_threshold,
        "top_features_per_concept": {
            k: [(int(idx), float(score)) for idx, score in v]
            for k, v in top_features.items()
        },
        "matching_features": {
            k: [(int(idx), float(r), float(p)) for idx, r, p in v]
            for k, v in matching.items()
        },
        "n_matching_per_concept": {k: len(v) for k, v in matching.items()},
    }

    print("\n=== Experiment 4: SAE Feature Discovery ===")
    for concept in top_features:
        n_match = results["n_matching_per_concept"].get(concept, 0)
        top = top_features[concept][:3]
        top_str = ", ".join(f"feat{idx}({score:.3f})" for idx, score in top)
        print(f"  {concept}: {n_match} features with |r|>={r_threshold}  top3: {top_str}")

    out_path = os.path.join(output_dir, "experiment4_sae.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", default="outputs/checkpoints/sae.pt")
    parser.add_argument("--activations", default="data/processed/activations_impala.h5")
    parser.add_argument("--probing", default="outputs/tables/experiment1_probing.json")
    parser.add_argument("--layer", default="embedder.fc")
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--l1_coef", type=float, default=0.01)
    parser.add_argument("--r_threshold", type=float, default=0.7)
    parser.add_argument("--output_dir", default="outputs/tables")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_sae_analysis(
        sae_path=args.sae,
        activations_path=args.activations,
        probing_path=args.probing,
        layer=args.layer,
        expansion_factor=args.expansion_factor,
        l1_coef=args.l1_coef,
        r_threshold=args.r_threshold,
        output_dir=args.output_dir,
    )
