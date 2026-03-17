"""Experiment 1, Step 1: Layer-wise contextual probing (D1 — contextual encoding).

Trains linear probes at each layer and reports in-context vs out-of-context accuracy.
High in-context, low out-of-context = evidence of contextual encoding.

Usage:
    uv run python run/pipeline/analysis/experiment1_probing.py

Prerequisites:
    - run/pipeline/prepare_data/collect_activations.py must have completed
    - Activations stored in data/processed/activations_impala.h5
"""
import gc
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _ram_mb() -> str:
    """Return current process RSS and system available RAM in MB."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024 ** 2
        avail_mb = psutil.virtual_memory().available / 1024 ** 2
        return f"RSS={rss_mb:.0f} MB  avail={avail_mb:.0f} MB"
    except ImportError:
        # Fallback: read /proc/meminfo (Linux)
        try:
            with open("/proc/meminfo") as f:
                lines = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
            avail_kb = lines.get("MemAvailable", 0)
            return f"avail={avail_kb // 1024} MB"
        except Exception:
            return "(RAM unavailable)"


def _discover_keys(path: str):
    """Return (layer_keys, label_keys) by inspecting the first HDF5 sample."""
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

    with HDF5ActivationDataset(path, mode="r") as ds:
        if len(ds) == 0:
            raise ValueError(f"Dataset at {path} is empty.")
        sample0 = ds[0]
        return list(sample0["activations"].keys()), list(sample0["labels"].keys())


def _load_labels(path: str, label_keys: list) -> dict:
    """Load all concept labels into memory (small — one float per step)."""
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset

    with HDF5ActivationDataset(path, mode="r") as ds:
        return {k: ds.get_all_labels(k) for k in label_keys}


def run_probing(
    activations_path: str = "data/processed/activations_impala.h5",
    probe_type: str = "linear",
    n_trials: int = 5,
    context_threshold: float = 0.5,
    output_dir: str = "outputs/tables",
) -> dict:
    """Run layer-wise contextual probing and save results.

    Loads one layer at a time to avoid loading the full activation tensor
    (10 convolutional layers × 50 k steps ≈ 15 GB) into memory simultaneously.
    """
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
    from src.probe_module.evaluator.probe_evaluator import ProbeEvaluator
    from src.analysis_module.visualization.shard_visualizer import ShardVisualizer

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layer_keys, label_keys = _discover_keys(activations_path)
    print(f"[INIT] Layers: {layer_keys}")
    print(f"[INIT] Concepts: {label_keys}")
    print(f"[INIT] RAM before label load: {_ram_mb()}")

    labels_by_concept = _load_labels(activations_path, label_keys)
    print(f"[INIT] RAM after label load:  {_ram_mb()}")

    evaluator = ProbeEvaluator(probe_name=probe_type, n_trials=n_trials)
    all_results = {}

    # Per-concept binarization.
    # The generic threshold=0.5 only makes sense for binary labelers (cheese_presence).
    # Distance labelers (proximity) need an inverted threshold: small distance = near = 1.
    # Direction labeler (angle in radians): positive angle means cheese is above the agent.
    _BINARIZERS = {
        "cheese_presence":  lambda v: (v > 0.5).astype(float),
        "cheese_proximity": lambda v: (v < 5.0).astype(float),   # 1 = cheese within 5 steps
        "cheese_direction": lambda v: (v > 0.0).astype(float),   # 1 = cheese above agent (dy>0)
        "corner_proximity": lambda v: (v < 5.0).astype(float),   # 1 = agent within 5 steps of corner
    }
    _default_binarizer = lambda v: (v > context_threshold).astype(float)
    binary_labels = {
        concept: _BINARIZERS.get(concept, _default_binarizer)(labels)
        for concept, labels in labels_by_concept.items()
    }

    # Context masks must be defined by a DIFFERENT signal than the probe target.
    # We use cheese_proximity as the behavioral context for all cheese-related concepts
    # (cheese nearby → cheese shard likely active) and corner_proximity for corner concept.
    # This tests: "Is concept X more decodable when the relevant shard is behaviorally active?"
    _cheese_ctx = (labels_by_concept.get("cheese_proximity", np.zeros(1)) < 5.0).astype(bool)
    _corner_ctx = (labels_by_concept.get("corner_proximity", np.zeros(1)) < 5.0).astype(bool)
    _CONTEXT_MASKS = {
        "cheese_presence":  _cheese_ctx,
        "cheese_proximity": _cheese_ctx,
        "cheese_direction": _cheese_ctx,
        "corner_proximity": _corner_ctx,
    }
    context_masks = {
        concept: _CONTEXT_MASKS.get(concept, _cheese_ctx)
        for concept in binary_labels
    }

    n_layers = len(layer_keys)
    # Process one layer at a time so only ~1-3 GB is live at once instead of ~15 GB
    layer_results_all: dict = {concept: {} for concept in labels_by_concept}

    with HDF5ActivationDataset(activations_path, mode="r") as ds:
        for i, layer in enumerate(layer_keys, 1):
            print(f"\n[LAYER {i}/{n_layers}] Loading '{layer}' — {_ram_mb()}")
            acts = ds.get_all_activations(layer)
            size_mb = acts.nbytes / 1024 ** 2
            print(f"[LAYER {i}/{n_layers}] Loaded shape={acts.shape} dtype={acts.dtype} size={size_mb:.0f} MB — {_ram_mb()}")

            for concept in labels_by_concept:
                print(f"  [PROBE] layer='{layer}' concept='{concept}' — {_ram_mb()}")
                result = evaluator.evaluate_context_split(
                    acts, binary_labels[concept], context_masks[concept]
                )
                print(f"  [PROBE] done — {_ram_mb()}")
                layer_results_all[concept][layer] = result

            print(f"[LAYER {i}/{n_layers}] Freeing '{layer}' — {_ram_mb()}")
            del acts
            gc.collect()
            print(f"[LAYER {i}/{n_layers}] Freed — {_ram_mb()}")

    all_results = layer_results_all

    for concept, layer_results in all_results.items():
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
