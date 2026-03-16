import numpy as np
import torch
from typing import Dict, List, Protocol, Tuple

from src.causal_module.patch.activation_patcher import ActivationPatcher


class AgentProtocol(Protocol):
    """Minimal interface required by CausalTracer — satisfied by ImpalaAgent."""

    def act_with_activations(
        self, obs: np.ndarray
    ) -> Tuple[int, Dict[str, torch.Tensor]]: ...

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray: ...

    @property
    def policy(self) -> torch.nn.Module: ...


class CausalTracer:
    """Implements causal tracing for RL policy networks (Meng et al., 2022).

    Protocol:
    1. Clean run: collect activations with concept present (e.g., cheese visible)
    2. Corrupted run: run with concept absent (no cheese), record baseline action probs
    3. Restoration: for each layer, restore clean activations during corrupted run,
       measure how much the target action probability recovers.

    A layer with high recovery score is causally important for the behavior.
    """

    def __init__(self, agent: AgentProtocol, layer_names: List[str]):
        self.agent = agent
        self.layer_names = layer_names
        self.patcher = ActivationPatcher(agent.policy)

    def trace(
        self,
        clean_obs: np.ndarray,
        corrupted_obs: np.ndarray,
        target_action_idx: int,
    ) -> Dict[str, float]:
        """For each layer, measure how much restoring clean activations recovers behavior.

        Returns:
            {layer: causal_effect_score}
            Score = (restored_prob - corrupted_prob) / (clean_prob - corrupted_prob + eps)
            Score of 1.0 = full recovery, 0.0 = no effect.
        """
        # Step 1: clean run — collect activations
        _, clean_acts = self.agent.act_with_activations(clean_obs)
        clean_prob = float(self.agent.get_action_probs(clean_obs)[target_action_idx])

        # Step 2: corrupted run baseline
        corrupted_prob = float(self.agent.get_action_probs(corrupted_obs)[target_action_idx])

        gap = clean_prob - corrupted_prob + 1e-8

        # Step 3: restoration per layer
        results: Dict[str, float] = {}
        for layer in self.layer_names:
            clean_act = clean_acts.get(layer)
            if clean_act is None:
                results[layer] = 0.0
                continue

            restore_patch = self.patcher.make_restore_patch(clean_act.unsqueeze(0))
            with self.patcher.patch_layer(layer, restore_patch):
                restored_prob = float(
                    self.agent.get_action_probs(corrupted_obs)[target_action_idx]
                )

            results[layer] = (restored_prob - corrupted_prob) / gap

        return results
