"""Experiment 1, Step 3: Integrate probe (D1) + causal (D2) results.

Cross-validates probe-detected layers with causally important layers
to produce a ranked list of shard candidates.

Usage:
    uv run python run/pipeline/analysis/experiment1_integrate.py

Prerequisites:
    - outputs/tables/experiment1_probing.json (from experiment1_probing.py)
    - outputs/tables/experiment1_causal.json (from experiment1_causal.py)
"""
import json
import os
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def integrate_results(
    probing_path: str = "outputs/tables/experiment1_probing.json",
    causal_path: str = "outputs/tables/experiment1_causal.json",
    output_dir: str = "outputs/tables",
    probe_threshold: float = 0.2,
    causal_threshold: float = 0.3,
    top_k: int = 3,
) -> dict:
    """Combine D1 (probe) and D2 (causal) to identify shard candidates."""
    from src.shard_module.detection.shard_detector import ShardDetector

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    probing = load_json(probing_path)
    causal = load_json(causal_path)

    detector = ShardDetector(
        probe_threshold=probe_threshold,
        causal_threshold=causal_threshold,
    )

    all_candidates = {}
    for concept in probing:
        probe_results = {
            layer: {
                "in_context_acc": metrics.get("in_context_acc", 0.0),
                "out_context_acc": metrics.get("out_context_acc", 0.0),
            }
            for layer, metrics in probing[concept].items()
        }
        candidates = detector.detect(probe_results, causal, top_k=top_k)
        all_candidates[concept] = [
            {"layer": layer, **scores} for layer, scores in candidates
        ]

    print("\n=== Shard Candidates ===")
    for concept, candidates in all_candidates.items():
        print(f"\nConcept: {concept}")
        for c in candidates:
            print(f"  {c['layer']}: probe_spec={c['probe_specificity']:.3f}, "
                  f"causal={c['causal_effect']:.3f}, combined={c['combined_score']:.3f}")

    output_path = os.path.join(output_dir, "experiment1_candidates.json")
    with open(output_path, "w") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"\nSaved candidates to {output_path}")
    return all_candidates


if __name__ == "__main__":
    integrate_results()
