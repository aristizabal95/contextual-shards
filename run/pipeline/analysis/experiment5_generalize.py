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
    - MiniGrid agent registered as "minigrid" in AgentFactory (for D2 causal tracing)
"""
import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_causal_tracing(
    checkpoint_path: str,
    causal_out: str,
    layer_names: list,
    n_trials: int = 50,
) -> dict:
    """Attempt D2 causal tracing with MiniGrid agent.

    Returns causal effects dict {layer: score} or empty dict on failure.
    Requires a MiniGrid agent registered as "minigrid" in AgentFactory that
    satisfies AgentProtocol (act_with_activations, get_action_probs, policy).
    """
    from src.agent_module import AgentFactory
    from src.environment_module import EnvFactory
    from src.causal_module.tracing.causal_tracer import CausalTracer

    _ckpt = checkpoint_path
    _layers = layer_names

    class _Cfg:
        class environment:
            name = "minigrid"
            env_id = "MiniGrid-Empty-5x5-v0"
            seed = 0
            num_levels = 500
        class agent:
            name = "minigrid"
            checkpoint_path = _ckpt
            layer_names = _layers

    cfg = _Cfg()
    causal_agent = AgentFactory("minigrid")(cfg)  # type: ignore[call-arg]
    causal_env = EnvFactory("minigrid")(cfg)  # type: ignore[call-arg]
    tracer = CausalTracer(causal_agent, layer_names)  # type: ignore[arg-type]

    causal_effects: dict = {}
    for _ in range(n_trials):
        clean_obs = causal_env.reset()
        corrupted_obs = causal_env.reset()
        clean_action = causal_agent.act(clean_obs)
        effects = tracer.trace(clean_obs, corrupted_obs, target_action_idx=clean_action)
        for layer, eff in effects.items():
            causal_effects.setdefault(layer, []).append(eff)

    mean_effects = {
        layer: float(sum(v) / len(v)) if v else 0.0
        for layer, v in causal_effects.items()
    }
    with open(causal_out, "w") as f:
        json.dump(mean_effects, f, indent=2)
    return mean_effects


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

    # ── Step 2: Causal tracing (D2) ────────────────────────────────────────
    logger.info("Step 2: Attempting causal tracing (D2)")
    layer_names = list(
        next(iter(probing_results.values()), {}).keys() if probing_results else []
    )
    causal_results: dict = {}
    try:
        causal_results = _run_causal_tracing(
            checkpoint_path=checkpoint_path,
            causal_out=causal_out,
            layer_names=layer_names,
        )
        results["causal"] = causal_results
        logger.info(f"Causal results saved to {causal_out}")
    except Exception as e:
        logger.warning(
            f"Causal tracing (D2) skipped: {e}\n"
            "To enable D2: implement a MiniGrid agent satisfying AgentProtocol "
            "(act_with_activations, get_action_probs, policy) and register it "
            "via @register_agent('minigrid')."
        )
        results["causal"] = {}

    # ── Step 3: Shard candidate integration ────────────────────────────────
    logger.info("Step 3: Integrating D1+D2 results into shard candidates")
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
        # Use real D2 results when available; fall back to D1 specificity proxy
        if causal_results:
            causal_d2 = {layer: causal_results.get(layer, 0.0) for layer in layer_metrics}
        else:
            causal_d2 = {
                layer: m.get("in_context_acc", 0.0) - m.get("out_context_acc", 0.0)
                for layer, m in layer_metrics.items()
            }
        candidates = detector.detect(probe_input, causal_d2, top_k=3)
        all_candidates[concept] = [
            {"layer": layer, **scores} for layer, scores in candidates
        ]

    with open(candidates_out, "w") as f:
        json.dump(all_candidates, f, indent=2)
    results["candidates"] = all_candidates

    # ── Step 4: Shard separability (D3) ────────────────────────────────────
    logger.info("Step 4: Attempting shard separability (D3)")
    try:
        from run.pipeline.analysis.experiment2_separability import run_separability
        separability_results = run_separability(
            candidates_path=candidates_out,
            activations_path=activations_path,
            checkpoint_path=checkpoint_path,
            output_dir=str(Path(separability_out).parent),
        )
        # Rename to exp5 separability path
        default_sep = str(Path(separability_out).parent / "experiment2_separability.json")
        if os.path.exists(default_sep) and default_sep != separability_out:
            os.rename(default_sep, separability_out)
        results["separability"] = separability_results
        logger.info(f"Separability results saved to {separability_out}")
    except Exception as e:
        logger.warning(
            f"Separability (D3) skipped: {e}\n"
            "run_separability currently targets the IMPALA/maze environment. "
            "Implement a MiniGrid-compatible version to enable D3 for Experiment 5."
        )
        results["separability"] = {}

    # ── Step 5: Print summary ───────────────────────────────────────────────
    print("\n=== Experiment 5: MiniGrid Generalization ===")
    print(f"  Activations: {activations_path}")
    for concept, candidates in all_candidates.items():
        print(f"\n  Concept: {concept}")
        for c in candidates[:3]:
            print(f"    {c['layer']}: combined={c.get('combined_score', 0):.3f}")

    if results.get("causal"):
        print(f"\n  D2 causal tracing: complete ({len(causal_results)} layers)")
    else:
        print("\n  D2 causal tracing: skipped (no MiniGrid agent with AgentProtocol)")

    if results.get("separability"):
        sep = results["separability"]
        print(f"\n  D3 separability: I(cheese→corner)={sep.get('I_cheese_corner', 'N/A'):.3f}, "
              f"I(corner→cheese)={sep.get('I_corner_cheese', 'N/A'):.3f}")
    else:
        print("\n  D3 separability: skipped")

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
