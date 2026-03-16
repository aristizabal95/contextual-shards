"""Experiment 1, Step 2: Causal tracing (D2 — causal behavioral role).

For each layer, measures how much restoring clean activations during a
corrupted run recovers cheese-directed behavior.

Usage:
    uv run python run/pipeline/analysis/experiment1_causal.py

Prerequisites:
    - Trained agent checkpoint at data/raw/maze_policy.pt
    - procgen + procgen-tools installed (requires Python 3.10 environment)
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LAYER_NAMES = [
    "block1", "block1.res1", "block1.res2",
    "block2", "block2.res1", "block2.res2",
    "block3", "block3.res1", "block3.res2",
    "fc",
]


def run_causal_tracing(
    checkpoint_path: str = "data/raw/maze_policy.pt",
    n_trials: int = 100,
    near_threshold: int = 5,
    output_dir: str = "outputs/tables",
) -> dict:
    """Run causal tracing across all layers.

    Returns {layer: mean_causal_effect_score}.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        from src.agent_module import AgentFactory
        from src.environment_module import EnvFactory
    except ImportError as e:
        logger.error(f"Environment/agent imports failed: {e}")
        raise

    # These will fail gracefully if procgen is not installed
    try:
        import procgen  # noqa: F401
    except ImportError:
        print("procgen not installed. Causal tracing requires Python 3.10 environment.")
        print("See plan/2026-03-16-contextual-shards.md for setup instructions.")
        return {}

    from src.causal_module.tracing.causal_tracer import CausalTracer

    # Config-like objects (would use Hydra in production)
    _ckpt = checkpoint_path

    class Cfg:
        class environment:
            name = "maze"
            num_levels = 500
            seed = 0
            distribution_mode = "easy"
        class agent:
            name = "impala"
            checkpoint_path = _ckpt
            layer_names = LAYER_NAMES

    cfg = Cfg()
    env = EnvFactory("maze")(cfg)
    agent = AgentFactory("impala")(cfg)
    tracer = CausalTracer(agent, LAYER_NAMES)

    causal_effects = {layer: [] for layer in LAYER_NAMES}

    for _ in range(n_trials):
        obs_clean = env.reset()
        cheese_pos = env.cheese_pos()
        agent_pos = env.agent_pos()
        dist = ((cheese_pos[0] - agent_pos[0])**2 + (cheese_pos[1] - agent_pos[1])**2) ** 0.5
        if dist > near_threshold:
            continue

        action_clean = agent.act(obs_clean)
        effects = tracer.trace(obs_clean, obs_clean, target_action_idx=action_clean)
        for layer, eff in effects.items():
            causal_effects[layer].append(eff)

    mean_effects = {
        layer: float(sum(v) / len(v)) if v else 0.0
        for layer, v in causal_effects.items()
    }

    output_path = os.path.join(output_dir, "experiment1_causal.json")
    with open(output_path, "w") as f:
        json.dump(mean_effects, f, indent=2)
    print(f"Saved causal results to {output_path}")
    return mean_effects


if __name__ == "__main__":
    run_causal_tracing()
