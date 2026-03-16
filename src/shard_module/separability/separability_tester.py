"""Shard separability testing — D3: quasi-independence of shards."""
import logging
from typing import Dict, List
import numpy as np
import torch

from src.causal_module.patch.activation_patcher import ActivationPatcher
from src.causal_module.shard_vector.shard_vector import ShardVector
from src.shard_module.metrics.shard_metrics import ShardMetrics

logger = logging.getLogger(__name__)


class SeparabilityTester:
    """Tests D3: suppressing shard S1 should not substantially affect shard S2.

    Protocol per research plan:
    1. Compute cheese shard vector and corner shard vector (mean activation diffs)
    2. Suppress cheese shard → measure corner-directed behavior (and vice versa)
    3. Compute independence score I(cheese, corner) and I(corner, cheese)
    4. Target: I > 0.8 for both directions

    Usage:
        tester = SeparabilityTester(agent=agent, shard_layer="fc")
        results = tester.run(...)
    """

    def __init__(self, agent, shard_layer: str):
        """
        Args:
            agent: any object with .policy (nn.Module) and .get_action_probs(obs) -> np.ndarray
            shard_layer: layer name to apply shard suppression (from Exp 1 candidates)
        """
        self.agent = agent
        self.shard_layer = shard_layer
        self.patcher = ActivationPatcher(agent.policy)
        self.sv = ShardVector()
        self.metrics = ShardMetrics()

    def _mean_action_prob(
        self,
        observations: List[np.ndarray],
        action_indices: List[int],
    ) -> float:
        """Mean probability of concept-directed actions across observations."""
        probs = []
        for obs, idx in zip(observations, action_indices):
            p = float(self.agent.get_action_probs(obs)[idx])
            probs.append(p)
        return float(np.mean(probs))

    def _suppressed_mean_action_prob(
        self,
        shard_vector: torch.Tensor,
        observations: List[np.ndarray],
        action_indices: List[int],
    ) -> float:
        """Mean action probability after projecting out shard direction."""
        patch_fn = self.patcher.make_suppress_patch(shard_vector)
        probs = []
        with self.patcher.patch_layer(self.shard_layer, patch_fn):
            for obs, idx in zip(observations, action_indices):
                p = float(self.agent.get_action_probs(obs)[idx])
                probs.append(p)
        return float(np.mean(probs))

    def run(
        self,
        cheese_context_acts: Dict[str, torch.Tensor],
        cheese_baseline_acts: Dict[str, torch.Tensor],
        corner_context_acts: Dict[str, torch.Tensor],
        corner_baseline_acts: Dict[str, torch.Tensor],
        cheese_obs: List[np.ndarray],
        cheese_actions: List[int],
        corner_obs: List[np.ndarray],
        corner_actions: List[int],
    ) -> Dict[str, float]:
        """Run separability test and return independence scores.

        Args:
            cheese_context_acts: activations when cheese is present
            cheese_baseline_acts: activations when cheese is absent
            corner_context_acts: activations when agent is near corner
            corner_baseline_acts: activations when agent is far from corner
            cheese_obs: observations for cheese-context behavior measurement
            cheese_actions: cheese-directed action indices per observation
            corner_obs: observations for corner-context behavior measurement
            corner_actions: corner-directed action indices per observation

        Returns:
            Dict with keys: I_cheese_corner, I_corner_cheese,
                           cheese_causal_effect, corner_causal_effect
        """
        # Compute shard vectors
        cheese_vecs = self.sv.compute(
            {self.shard_layer: cheese_context_acts[self.shard_layer]},
            {self.shard_layer: cheese_baseline_acts[self.shard_layer]},
        )
        corner_vecs = self.sv.compute(
            {self.shard_layer: corner_context_acts[self.shard_layer]},
            {self.shard_layer: corner_baseline_acts[self.shard_layer]},
        )
        cheese_vec = cheese_vecs[self.shard_layer]
        corner_vec = corner_vecs[self.shard_layer]

        # Baseline behaviors
        cheese_baseline_beh = self._mean_action_prob(cheese_obs, cheese_actions)
        corner_baseline_beh = self._mean_action_prob(corner_obs, corner_actions)

        # Self-suppression (denominator for independence score)
        cheese_self_suppressed = self._suppressed_mean_action_prob(cheese_vec, cheese_obs, cheese_actions)
        corner_self_suppressed = self._suppressed_mean_action_prob(corner_vec, corner_obs, corner_actions)

        # Cross-suppression (numerator for independence score)
        cheese_suppress_corner = self._suppressed_mean_action_prob(cheese_vec, corner_obs, corner_actions)
        corner_suppress_cheese = self._suppressed_mean_action_prob(corner_vec, cheese_obs, cheese_actions)

        return {
            "I_cheese_corner": self.metrics.independence_score(
                corner_baseline_beh, cheese_suppress_corner, corner_self_suppressed
            ),
            "I_corner_cheese": self.metrics.independence_score(
                cheese_baseline_beh, corner_suppress_cheese, cheese_self_suppressed
            ),
            "cheese_causal_effect": self.metrics.causal_effect_size(cheese_baseline_beh, cheese_self_suppressed),
            "corner_causal_effect": self.metrics.causal_effect_size(corner_baseline_beh, corner_self_suppressed),
        }
