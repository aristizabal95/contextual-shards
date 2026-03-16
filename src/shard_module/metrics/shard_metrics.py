"""Metrics for shard theory experiments."""


class ShardMetrics:
    """Computes shard-theory specific evaluation metrics.

    Key metrics from the research plan:
    - independence_score: measures D3 (separability) — how much suppressing S1 leaks into S2
    - causal_effect_size: measures D2 — Δaction probability from shard suppression
    """

    def independence_score(
        self,
        baseline_behavior_S2: float,
        suppressed_behavior_S2: float,
        suppressed_behavior_self: float,
    ) -> float:
        """Compute I(S1, S2): independence of S2 from S1 suppression.

        I = 1 − |Δbehavior(S2) when S1 suppressed| / |Δbehavior(S2) when S2 suppressed|

        Interpretation:
          I = 1.0 → full independence (suppressing S1 has no effect on S2 behavior)
          I = 0.0 → full dependence (suppressing S1 disrupts S2 as much as suppressing S2 itself)
          I < 0.0 → suppressing S1 amplifies S2 behavior (unexpected interaction)

        Args:
            baseline_behavior_S2: S2-concept-directed behavior probability (no suppression)
            suppressed_behavior_S2: S2 behavior after S1 is suppressed
            suppressed_behavior_self: S2 behavior after S2 itself is suppressed (denominator)

        Returns:
            Independence score I ∈ (-∞, 1.0], target > 0.8 per research plan
        """
        leakage = abs(baseline_behavior_S2 - suppressed_behavior_S2)
        self_effect = abs(baseline_behavior_S2 - suppressed_behavior_self) + 1e-8
        return float(1.0 - leakage / self_effect)

    def causal_effect_size(
        self,
        baseline_prob: float,
        intervened_prob: float,
    ) -> float:
        """Compute |Δaction probability| from shard suppression/activation.

        Args:
            baseline_prob: action probability without intervention
            intervened_prob: action probability after suppressing shard vector

        Returns:
            Absolute change in action probability (target > 0.3 per research plan)
        """
        return float(abs(baseline_prob - intervened_prob))
