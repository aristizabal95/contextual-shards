"""Experiment 1, Step 1: Layer-wise contextual probing (D1 — contextual encoding).

Trains linear probes at each layer and reports in-context vs out-of-context accuracy.
High in-context, low out-of-context = evidence of contextual encoding.

Usage:
    uv run python run/pipeline/analysis/experiment1_probing.py

Prerequisites:
    - run/pipeline/prepare_data/collect_activations.py must have completed
    - Activations stored in data/processed/activations_impala.h5
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dataset(path: str):
    """Load all activations and labels from HDF5 dataset into memory."""
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

    with HDF5ActivationDataset(path, mode="r") as ds:
        n = len(ds)
        if n == 0:
            raise ValueError(f"Dataset at {path} is empty.")

        # Collect first sample to discover keys
        sample0 = ds[0]
        layer_keys = list(sample0["activations"].keys())
        label_keys = list(sample0["labels"].keys())

        acts_by_layer = {k: ds.get_all_activations(k) for k in layer_keys}
        labels_by_concept = {k: ds.get_all_labels(k) for k in label_keys}

    return acts_by_layer, labels_by_concept


def run_probing(
    activations_path: str = "data/processed/activations_impala.h5",
    probe_type: str = "linear",
    n_trials: int = 5,
    context_threshold: float = 0.5,
    output_dir: str = "outputs/tables",
) -> dict:
    """Run layer-wise contextual probing and save results."""
    from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
    from src.analysis_module.visualization.shard_visualizer import ShardVisualizer

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    acts_by_layer, labels_by_concept = load_dataset(activations_path)

    evaluator = ProbeEvaluator(probe_name=probe_type, n_trials=n_trials)
    all_results = {}

    for concept, labels in labels_by_concept.items():
        y_binary = (labels > context_threshold).astype(float)
        context_mask = y_binary.astype(bool)
        layer_results = evaluator.evaluate_all_layers(acts_by_layer, y_binary, context_mask)
        all_results[concept] = layer_results

        print(f"\n=== {concept} ===")
        for layer, metrics in layer_results.items():
            in_acc = metrics.get("in_context_acc", float("nan"))
            out_acc = metrics.get("out_context_acc", float("nan"))
            print(f"  {layer}: in={in_acc:.3f}, out={out_acc:.3f}")

    output_path = os.path.join(output_dir, "experiment1_probing.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    try:
        viz = ShardVisualizer()
        fig = viz.plot_probe_heatmap(all_results, output_path="outputs/figures/exp1_probe_heatmap.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Visualization failed (non-fatal): {e}")

    return all_results


if __name__ == "__main__":
    run_probing()
