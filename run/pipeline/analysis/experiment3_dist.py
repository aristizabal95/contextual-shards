"""Experiment 3: Reinforcement distribution correlation analysis (D4).

Tests whether corner shard strength (from Experiment 1 causal results) correlates
with the frequency of corner reinforcement during training.

Requires three agents trained with different cheese distributions:
  - corner_biased:  75% cheese in top-right corner  (freq = 0.75)
  - uniform:        ~25% cheese in top-right         (freq = 0.25)
  - anti_corner:    10% cheese in top-right          (freq = 0.10)

Usage (after running Experiment 1 on each agent's activations):
    uv run python run/pipeline/analysis/experiment3_dist.py \
        --causal_dir outputs/tables

Each agent's causal results must be saved as:
    outputs/tables/experiment1_causal_corner_biased.json
    outputs/tables/experiment1_causal_uniform.json
    outputs/tables/experiment1_causal_anti_corner.json

Or pass a comma-separated list with --agents.
"""
import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Known corner-reinforcement frequencies for each distribution
REINFORCEMENT_FREQS = {
    "corner_biased": 0.75,
    "uniform": 0.25,
    "anti_corner": 0.10,
}

CORNER_LAYER = "embedder.fc"   # layer to read corner shard strength from


def load_corner_strength(causal_path: str, layer: str = CORNER_LAYER) -> float:
    """Extract corner shard causal effect from an experiment1_causal.json file."""
    with open(causal_path) as f:
        data = json.load(f)
    if layer not in data:
        available = list(data.keys())
        logger.warning(f"Layer '{layer}' not in {causal_path}. Available: {available}")
        layer = available[-1] if available else layer
    return float(data[layer])


def run_correlation(
    causal_dir: str = "outputs/tables",
    agents: str = "corner_biased,uniform,anti_corner",
    layer: str = CORNER_LAYER,
    output_dir: str = "outputs/tables",
) -> dict:
    from src.analysis_module.statistics.stat_tests import report_experiment3_correlation

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    agent_names = [a.strip() for a in agents.split(",")]
    missing = []
    freqs, strengths = [], []

    for name in agent_names:
        path = os.path.join(causal_dir, f"experiment1_causal_{name}.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        strength = load_corner_strength(path, layer)
        freq = REINFORCEMENT_FREQS.get(name, 0.0)
        freqs.append(freq)
        strengths.append(strength)
        logger.info(f"  {name}: freq={freq:.2f}, shard_strength={strength:.4f}")

    if missing:
        print("\nMissing causal result files:")
        for p in missing:
            print(f"  {p}")
        print("\nTo generate them:")
        print("  1. Train each agent:  train_rl_agent.py --distribution <name>")
        print("  2. Collect activations: collect_activations.py --checkpoint outputs/checkpoints/<name>.pt \\")
        print("                           --output data/processed/activations_<name>.h5")
        print("  3. Run causal tracing: experiment1_causal.py")
        print("     (save output as outputs/tables/experiment1_causal_<name>.json)")
        if len(strengths) < 3:
            print(f"\nOnly {len(strengths)}/3 agents available. Cannot compute correlation yet.")
            return {"status": "incomplete", "missing": missing}

    result = report_experiment3_correlation(
        agent_names=agent_names[:len(strengths)],
        corner_reinforcement_freqs=freqs,
        corner_shard_strengths=strengths,
    )

    print("\n=== Experiment 3: D4 Reinforcement Traceability ===")
    print(f"  Spearman rho:  {result['spearman_rho']:.3f}")
    print(f"  p-value:       {result['p_value']:.4f}")
    print(f"  Meets target:  {'✓' if result['meets_target'] else '✗'}  (target: rho > 0.7)")
    for name, freq, strength in zip(agent_names, freqs, strengths):
        print(f"  {name:20s}: freq={freq:.2f}, strength={strength:.4f}")

    out_path = os.path.join(output_dir, "experiment3_correlation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal_dir", default="outputs/tables")
    parser.add_argument("--agents", default="corner_biased,uniform,anti_corner")
    parser.add_argument("--layer", default=CORNER_LAYER)
    parser.add_argument("--output_dir", default="outputs/tables")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_correlation(args.causal_dir, args.agents, args.layer, args.output_dir)
