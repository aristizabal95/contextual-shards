"""Experiment 2: Shard separability test (D3 — quasi-independence of shards).

Tests whether suppressing the cheese shard direction affects corner behavior
and vice versa. Independence score I > 0.8 for both directions is the target.

Usage (after Experiment 1 has run):
    uv run python run/pipeline/analysis/experiment2_separability.py

Prerequisites:
    - outputs/tables/experiment1_candidates.json (from experiment1_integrate.py)
    - data/processed/activations_impala.h5 (from collect_activations.py)
    - data/raw/maze_policy.pt
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

LAYER_NAMES = [
    "embedder.block1", "embedder.block1.res1", "embedder.block1.res2",
    "embedder.block2", "embedder.block2.res1", "embedder.block2.res2",
    "embedder.block3", "embedder.block3.res1", "embedder.block3.res2",
    "embedder.fc",
]

NEAR_THRESHOLD = 5


def load_shard_layer(candidates_path: str) -> str:
    """Pick the top-scoring layer from Experiment 1 candidates."""
    with open(candidates_path) as f:
        candidates = json.load(f)
    best_layer, best_score = None, -1.0
    for concept_candidates in candidates.values():
        for c in concept_candidates:
            if c["combined_score"] > best_score:
                best_score = c["combined_score"]
                best_layer = c["layer"]
    if best_layer is None:
        logger.warning("No candidates found; defaulting to embedder.fc")
        best_layer = "embedder.fc"
    return best_layer


def split_activations_by_label(
    activations: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (context_acts, baseline_acts) split by label threshold."""
    mask = labels > threshold
    return activations[mask], activations[~mask]


def collect_context_obs(
    env,
    n: int,
    near_fn: Callable,
    max_tries: int = 2000,
) -> List[np.ndarray]:
    """Collect n observations where near_fn(env) returns True after env.reset()."""
    obs_list = []
    for _ in range(max_tries):
        if len(obs_list) >= n:
            break
        obs = env.reset()
        if near_fn(env):
            obs_list.append(obs)
    if not obs_list:
        raise RuntimeError(
            f"Could not collect any context observations in {max_tries} env resets"
        )
    return obs_list


def measure_behavioral_change(
    model: torch.nn.Module,
    patcher,
    suppress_vec: torch.Tensor,
    layer_name: str,
    obs_list: List[np.ndarray],
    device: torch.device,
) -> Tuple[float, float]:
    """Return (mean_baseline_prob, mean_suppressed_prob) for greedy actions.

    For each observation, the behavioral measure is P(greedy action without suppression).
    We compare this probability before and after projecting out suppress_vec from the
    layer activations. A large drop indicates the suppressed direction is causally
    relevant to concept-directed behavior.
    """
    suppress_fn = patcher.make_suppress_patch(suppress_vec)
    baseline_probs, suppressed_probs = [], []
    for obs in obs_list:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            dist_base, _ = model(obs_t)
            probs_base = dist_base.probs[0].cpu()
        greedy = int(probs_base.argmax().item())
        baseline_probs.append(float(probs_base[greedy]))
        with patcher.patch_layer(layer_name, suppress_fn):
            with torch.no_grad():
                dist_supp, _ = model(obs_t)
                probs_supp = dist_supp.probs[0].cpu()
        suppressed_probs.append(float(probs_supp[greedy]))
    return float(np.mean(baseline_probs)), float(np.mean(suppressed_probs))


def run_separability(
    candidates_path: str = "outputs/tables/experiment1_candidates.json",
    activations_path: str = "data/processed/activations_impala.h5",
    checkpoint_path: str = "data/raw/maze_policy.pt",
    n_obs: int = 64,
    output_dir: str = "outputs/tables",
) -> Dict:
    from src.data_module.activation_dataset.activation_dataset import HDF5ActivationDataset
    from src.causal_module.shard_vector.shard_vector import ShardVector
    from src.causal_module.patch.activation_patcher import ActivationPatcher
    from src.shard_module.metrics.shard_metrics import ShardMetrics
    from src.agent_module.policy.impala_agent import ImpalaAgent
    from src.environment_module.maze.maze_env import ProcgenMazeEnv

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shard_layer = load_shard_layer(candidates_path)
    logger.info(f"Using shard layer: {shard_layer}")

    _ckpt = checkpoint_path

    class Cfg:
        class agent:
            checkpoint_path = _ckpt
            layer_names = LAYER_NAMES
        class environment:
            num_levels = 500
            seed = 0
            distribution_mode = "easy"

    impala = ImpalaAgent(Cfg())
    model = impala.model
    assert model is not None
    device = next(model.parameters()).device
    patcher = ActivationPatcher(model)

    env = ProcgenMazeEnv(Cfg())

    def _cheese_near(e) -> bool:
        ap = e.agent_pos()
        cp = e.cheese_pos()
        return ((ap[0] - cp[0]) ** 2 + (ap[1] - cp[1]) ** 2) ** 0.5 < NEAR_THRESHOLD

    def _corner_near(e) -> bool:
        ap = e.agent_pos()
        try:
            from procgen_tools.maze import state_from_venv  # type: ignore[import]
            state = state_from_venv(e.venv, 0)
            w = state.inner_grid().shape[1]
        except Exception:
            w = 25
        return ap[0] <= NEAR_THRESHOLD and ap[1] >= w - 1 - NEAR_THRESHOLD

    logger.info(f"Collecting up to {n_obs} cheese-context observations...")
    cheese_ctx_obs = collect_context_obs(env, n_obs, _cheese_near)
    logger.info(f"Collecting up to {n_obs} corner-context observations...")
    corner_ctx_obs = collect_context_obs(env, n_obs, _corner_near)
    logger.info(
        f"Collected {len(cheese_ctx_obs)} cheese-context and "
        f"{len(corner_ctx_obs)} corner-context observations"
    )

    # Load activations and labels from HDF5 for shard vector computation
    with HDF5ActivationDataset(activations_path, mode="r") as ds:
        acts = ds.get_all_activations(shard_layer)
        cheese_labels = ds.get_all_labels("cheese_presence")
        corner_labels = ds.get_all_labels("corner_proximity")

    # corner_proximity is stored as distance; binarize to near=1, far=0
    corner_labels_bin = (corner_labels < NEAR_THRESHOLD).astype(float)

    cheese_ctx_acts, cheese_base_acts = split_activations_by_label(acts, cheese_labels)
    corner_ctx_acts, corner_base_acts = split_activations_by_label(acts, corner_labels_bin)

    sv = ShardVector()
    cheese_vecs = sv.compute(
        {shard_layer: torch.from_numpy(cheese_ctx_acts).float()},
        {shard_layer: torch.from_numpy(cheese_base_acts).float()},
    )
    corner_vecs = sv.compute(
        {shard_layer: torch.from_numpy(corner_ctx_acts).float()},
        {shard_layer: torch.from_numpy(corner_base_acts).float()},
    )
    cheese_vec = cheese_vecs[shard_layer].to(device)
    corner_vec = corner_vecs[shard_layer].to(device)
    logger.info(f"Cheese shard vector norm: {cheese_vec.norm():.4f}")
    logger.info(f"Corner shard vector norm: {corner_vec.norm():.4f}")

    # Corner-context measurements: does suppressing cheese disturb corner behavior?
    #   corner_self_supp: corner behavior when its OWN direction is suppressed (denominator)
    #   cheese_cross_supp: corner behavior when CHEESE direction is suppressed (cross effect)
    corner_base_beh, corner_self_supp = measure_behavioral_change(
        model, patcher, corner_vec, shard_layer, corner_ctx_obs, device
    )
    _, cheese_cross_supp = measure_behavioral_change(
        model, patcher, cheese_vec, shard_layer, corner_ctx_obs, device
    )

    # Cheese-context measurements: does suppressing corner disturb cheese behavior?
    #   cheese_self_supp: cheese behavior when its OWN direction is suppressed (denominator)
    #   corner_cross_supp: cheese behavior when CORNER direction is suppressed (cross effect)
    cheese_base_beh, cheese_self_supp = measure_behavioral_change(
        model, patcher, cheese_vec, shard_layer, cheese_ctx_obs, device
    )
    _, corner_cross_supp = measure_behavioral_change(
        model, patcher, corner_vec, shard_layer, cheese_ctx_obs, device
    )

    metrics = ShardMetrics()
    # I_cheese_corner: independence of corner shard from cheese shard suppression
    I_cheese_corner = metrics.independence_score(
        corner_base_beh, cheese_cross_supp, corner_self_supp
    )
    # I_corner_cheese: independence of cheese shard from corner shard suppression
    I_corner_cheese = metrics.independence_score(
        cheese_base_beh, corner_cross_supp, cheese_self_supp
    )

    results = {
        "shard_layer": shard_layer,
        "I_cheese_corner": I_cheese_corner,
        "I_corner_cheese": I_corner_cheese,
        "cheese_causal_effect": metrics.causal_effect_size(cheese_base_beh, cheese_self_supp),
        "corner_causal_effect": metrics.causal_effect_size(corner_base_beh, corner_self_supp),
        "target": 0.8,
        "cheese_meets_target": I_cheese_corner >= 0.8,
        "corner_meets_target": I_corner_cheese >= 0.8,
    }

    print("\n=== Experiment 2: Shard Separability (D3) ===")
    print(f"  Shard layer:      {shard_layer}")
    print(f"  I(cheese→corner): {I_cheese_corner:.3f}  {'✓' if results['cheese_meets_target'] else '✗'}")
    print(f"  I(corner→cheese): {I_corner_cheese:.3f}  {'✓' if results['corner_meets_target'] else '✗'}")
    print(f"  (target > 0.8)")

    out_path = os.path.join(output_dir, "experiment2_separability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/tables/experiment1_candidates.json")
    parser.add_argument("--activations", default="data/processed/activations_impala.h5")
    parser.add_argument("--checkpoint", default="data/raw/maze_policy.pt")
    parser.add_argument("--n_obs", type=int, default=64)
    parser.add_argument("--output_dir", default="outputs/tables")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_separability(
        args.candidates,
        args.activations,
        args.checkpoint,
        n_obs=args.n_obs,
        output_dir=args.output_dir,
    )
