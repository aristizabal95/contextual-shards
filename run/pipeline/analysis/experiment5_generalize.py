"""Experiment 5: Apply shard detection pipeline to a new environment (MiniGrid).

Runs the full Experiment 1+2 pipeline on a MiniGrid agent to test whether
the detection framework generalizes across environments.

Usage (after training a MiniGrid agent and collecting activations):
    uv run python run/pipeline/analysis/experiment5_generalize.py \
        --activations data/processed/activations_minigrid.h5 \
        --probing_out outputs/tables/exp5_probing.json \
        --causal_out outputs/tables/exp5_causal.json \
        --candidates_out outputs/tables/exp5_candidates.json

Prerequisites:
    - A trained MiniGrid agent checkpoint
    - Activations collected via collect_activations.py with environment=minigrid
"""
import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def run_generalization(
    activations_path: str = "data/processed/activations_minigrid.h5",
    probing_out: str = "outputs/tables/exp5_probing.json",
    causal_out: str = "outputs/tables/exp5_causal.json",
    candidates_out: str = "outputs/tables/exp5_candidates.json",
    separability_out: str = "outputs/tables/exp5_separability.json",
    checkpoint_path: str = "outputs/checkpoints/minigrid_agent.pt",
    probe_type: str = "linear",
    probe_threshold: float = 0.2,
    causal_threshold: float = 0.3,
) -> dict:
    """Run the full shard detection pipeline on MiniGrid activations."""

    if not os.path.exists(activations_path):
        raise FileNotFoundError(
            f"MiniGrid activations not found: {activations_path}\n"
            "Steps to generate them:\n"
            "  1. Train a MiniGrid agent (or adapt train_rl_agent.py)\n"
            "  2. Run: collect_activations.py --checkpoint <ckpt> --output "
            f"{activations_path}"
        )

    for path in [probing_out, causal_out, candidates_out, separability_out]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # ── Step 1: Linear probing (D1) ────────────────────────────────────────
    logger.info("Step 1: Running linear probing (D1)")
    from run.pipeline.analysis.experiment1_probing import run_probing
    probing_results = run_probing(
        activations_path=activations_path,
        probe_type=probe_type,
        output_dir=str(Path(probing_out).parent),
    )
    # Rename output to exp5 path
    default_path = str(Path(probing_out).parent / "experiment1_probing.json")
    if os.path.exists(default_path) and default_path != probing_out:
        os.rename(default_path, probing_out)
    results["probing"] = probing_results
    logger.info(f"Probing results saved to {probing_out}")

    # ── Step 2: Shard candidate integration ────────────────────────────────
    logger.info("Step 2: Integrating D1 results into shard candidates")
    from src.shard_module.detection.shard_detector import ShardDetector

    detector = ShardDetector(
        probe_threshold=probe_threshold,
        causal_threshold=causal_threshold,
    )
    all_candidates = {}
    for concept, layer_metrics in probing_results.items():
        probe_input = {
            layer: {
                "in_context_acc": m.get("in_context_acc", 0.0),
                "out_context_acc": m.get("out_context_acc", 0.0),
            }
            for layer, m in layer_metrics.items()
        }
        # No causal results for MiniGrid (no procgen dependency)
        # Use probe_specificity alone as combined score proxy
        causal_proxy = {
            layer: m.get("in_context_acc", 0.0) - m.get("out_context_acc", 0.0)
            for layer, m in layer_metrics.items()
        }
        candidates = detector.detect(probe_input, causal_proxy, top_k=3)
        all_candidates[concept] = [
            {"layer": layer, **scores} for layer, scores in candidates
        ]

    with open(candidates_out, "w") as f:
        json.dump(all_candidates, f, indent=2)
    results["candidates"] = all_candidates

    # ── Step 3: Print summary ───────────────────────────────────────────────
    print("\n=== Experiment 5: MiniGrid Generalization ===")
    print(f"  Activations: {activations_path}")
    for concept, candidates in all_candidates.items():
        print(f"\n  Concept: {concept}")
        for c in candidates[:3]:
            print(f"    {c['layer']}: combined={c.get('combined_score', 0):.3f}")

    print(f"\nCandidates saved to {candidates_out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="data/processed/activations_minigrid.h5")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/minigrid_agent.pt")
    parser.add_argument("--probing_out", default="outputs/tables/exp5_probing.json")
    parser.add_argument("--causal_out", default="outputs/tables/exp5_causal.json")
    parser.add_argument("--candidates_out", default="outputs/tables/exp5_candidates.json")
    parser.add_argument("--separability_out", default="outputs/tables/exp5_separability.json")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_generalization(
        activations_path=args.activations,
        probing_out=args.probing_out,
        causal_out=args.causal_out,
        candidates_out=args.candidates_out,
        separability_out=args.separability_out,
        checkpoint_path=args.checkpoint,
    )
