from src.shard_module.detection.shard_detector import ShardDetector
from src.shard_module.metrics.shard_metrics import ShardMetrics


def _make_probe_results():
    return {
        "block1": {"in_context_acc": 0.65, "out_context_acc": 0.52},
        "block2": {"in_context_acc": 0.80, "out_context_acc": 0.51},
        "block3": {"in_context_acc": 0.90, "out_context_acc": 0.50},
        "fc": {"in_context_acc": 0.95, "out_context_acc": 0.50},
    }


def _make_causal_results():
    return {
        "block1": 0.05,
        "block2": 0.20,
        "block3": 0.45,
        "fc": 0.70,
    }


class TestShardDetector:
    def test_compute_combined_scores_keys(self):
        detector = ShardDetector()
        scores = detector.compute_combined_scores(_make_probe_results(), _make_causal_results())
        assert set(scores.keys()) == {"block1", "block2", "block3", "fc"}

    def test_combined_score_formula(self):
        detector = ShardDetector()
        probe = {"fc": {"in_context_acc": 0.9, "out_context_acc": 0.5}}
        causal = {"fc": 0.6}
        scores = detector.compute_combined_scores(probe, causal)
        expected_spec = 0.9 - 0.5
        expected_combined = expected_spec * 0.6
        assert abs(scores["fc"]["probe_specificity"] - expected_spec) < 1e-5
        assert abs(scores["fc"]["combined_score"] - expected_combined) < 1e-5

    def test_get_top_candidates_filters_thresholds(self):
        detector = ShardDetector(probe_threshold=0.2, causal_threshold=0.3)
        scores = detector.compute_combined_scores(_make_probe_results(), _make_causal_results())
        candidates = detector.get_top_candidates(scores, top_k=5)
        # Only block3 and fc meet both thresholds
        candidate_layers = [c[0] for c in candidates]
        assert "fc" in candidate_layers
        assert "block3" in candidate_layers
        assert "block1" not in candidate_layers  # causal=0.05 < 0.3

    def test_get_top_candidates_sorted_descending(self):
        detector = ShardDetector(probe_threshold=0.0, causal_threshold=0.0)
        scores = detector.compute_combined_scores(_make_probe_results(), _make_causal_results())
        candidates = detector.get_top_candidates(scores, top_k=4, require_both_thresholds=False)
        combined_scores = [c[1]["combined_score"] for c in candidates]
        assert combined_scores == sorted(combined_scores, reverse=True)

    def test_detect_convenience_method(self):
        detector = ShardDetector(probe_threshold=0.3, causal_threshold=0.4)
        candidates = detector.detect(_make_probe_results(), _make_causal_results(), top_k=2)
        assert len(candidates) <= 2
        assert all("combined_score" in c[1] for c in candidates)

    def test_no_candidates_falls_back_gracefully(self):
        detector = ShardDetector(probe_threshold=0.99, causal_threshold=0.99)
        probe = {"block1": {"in_context_acc": 0.6, "out_context_acc": 0.5}}
        causal = {"block1": 0.1}
        scores = detector.compute_combined_scores(probe, causal)
        # No candidates meet thresholds — fallback returns top-k anyway
        candidates = detector.get_top_candidates(scores, top_k=1)
        assert len(candidates) == 1


class TestShardMetrics:
    def test_independence_score_full_independence(self):
        metrics = ShardMetrics()
        # S2 behavior unchanged when S1 suppressed
        I = metrics.independence_score(
            baseline_behavior_S2=0.8,
            suppressed_behavior_S2=0.8,  # no change
            suppressed_behavior_self=0.3,
        )
        assert abs(I - 1.0) < 1e-4

    def test_independence_score_full_dependence(self):
        metrics = ShardMetrics()
        # S2 behavior drops by same amount when S1 suppressed vs S2 suppressed
        I = metrics.independence_score(
            baseline_behavior_S2=0.8,
            suppressed_behavior_S2=0.3,  # same as self-suppression
            suppressed_behavior_self=0.3,
        )
        assert abs(I - 0.0) < 1e-4

    def test_independence_score_partial(self):
        metrics = ShardMetrics()
        I = metrics.independence_score(
            baseline_behavior_S2=0.8,
            suppressed_behavior_S2=0.65,  # partial leak
            suppressed_behavior_self=0.3,
        )
        # leakage = |0.8 - 0.65| = 0.15; self_effect = |0.8 - 0.3| = 0.5
        # I = 1 - 0.15/0.5 = 0.7
        assert abs(I - 0.7) < 1e-4

    def test_causal_effect_size(self):
        metrics = ShardMetrics()
        effect = metrics.causal_effect_size(0.8, 0.5)
        assert abs(effect - 0.3) < 1e-5

    def test_causal_effect_size_symmetric(self):
        metrics = ShardMetrics()
        assert metrics.causal_effect_size(0.3, 0.7) == metrics.causal_effect_size(0.7, 0.3)
