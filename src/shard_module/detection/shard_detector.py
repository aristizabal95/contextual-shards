"""Shard candidate detection by combining probe (D1) and causal (D2) results."""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ShardDetector:
    """Identifies shard candidates by intersecting probe and causal tracing results.

    A layer is a shard candidate if it satisfies both:
    - D1: probe specificity (in_context_acc − out_context_acc) >= probe_threshold
    - D2: causal effect >= causal_threshold

    Combined score = probe_specificity * causal_effect (higher = stronger candidate).
    """

    def __init__(
        self,
        probe_threshold: float = 0.2,
        causal_threshold: float = 0.3,
    ):
        self.probe_threshold = probe_threshold
        self.causal_threshold = causal_threshold

    def compute_combined_scores(
        self,
        probe_results: Dict[str, Dict[str, float]],
        causal_results: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute combined shard scores for each layer.

        Args:
            probe_results: {layer: {"in_context_acc": ..., "out_context_acc": ...}}
            causal_results: {layer: causal_effect_score}

        Returns:
            {layer: {"probe_specificity": ..., "causal_effect": ..., "combined_score": ...}}
        """
        scores: Dict[str, Dict[str, float]] = {}
        for layer, probe in probe_results.items():
            in_acc = probe.get("in_context_acc", 0.0)
            out_acc = probe.get("out_context_acc", 0.0)
            specificity = float(in_acc - out_acc)
            causal = float(causal_results.get(layer, 0.0))
            scores[layer] = {
                "probe_specificity": specificity,
                "causal_effect": causal,
                "combined_score": specificity * causal,
            }
        return scores

    def get_top_candidates(
        self,
        scores: Dict[str, Dict[str, float]],
        top_k: int = 3,
        require_both_thresholds: bool = True,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Return top-k shard candidates ranked by combined score.

        Args:
            scores: output of compute_combined_scores()
            top_k: number of candidates to return
            require_both_thresholds: if True, filter layers below D1/D2 thresholds first

        Returns:
            List of (layer_name, scores_dict) sorted descending by combined_score
        """
        candidates = list(scores.items())

        if require_both_thresholds:
            candidates = [
                (layer, s) for layer, s in candidates
                if s["probe_specificity"] >= self.probe_threshold
                and s["causal_effect"] >= self.causal_threshold
            ]
            if not candidates:
                logger.warning(
                    f"No candidates met thresholds (probe>={self.probe_threshold}, "
                    f"causal>={self.causal_threshold}). Returning top-k without filtering."
                )
                candidates = list(scores.items())

        candidates.sort(key=lambda x: -x[1]["combined_score"])
        return candidates[:top_k]

    def detect(
        self,
        probe_results: Dict[str, Dict[str, float]],
        causal_results: Dict[str, float],
        top_k: int = 3,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Convenience method: compute scores and return top candidates."""
        scores = self.compute_combined_scores(probe_results, causal_results)
        return self.get_top_candidates(scores, top_k=top_k)
