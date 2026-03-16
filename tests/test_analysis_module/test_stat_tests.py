import numpy as np
import pytest
from src.analysis_module.statistics.stat_tests import (
    bonferroni_correction,
    spearman_correlation,
    cohen_d,
    report_experiment3_correlation,
)
from src.analysis_module.visualization.shard_visualizer import ShardVisualizer
import matplotlib.pyplot as plt


def test_bonferroni_correction_basic():
    p_values = [0.001, 0.01, 0.05, 0.1]
    # threshold = 0.01 / 4 = 0.0025
    results = bonferroni_correction(p_values, alpha=0.01)
    assert results[0] is True    # 0.001 < 0.0025
    assert results[1] is False   # 0.01 >= 0.0025
    assert results[2] is False
    assert results[3] is False


def test_bonferroni_correction_empty():
    assert bonferroni_correction([]) == []


def test_bonferroni_all_significant():
    p_values = [0.0001] * 10
    results = bonferroni_correction(p_values, alpha=0.05)
    # threshold = 0.05 / 10 = 0.005; 0.0001 < 0.005 -> all True
    assert all(results)


def test_spearman_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, p = spearman_correlation(x, y)
    assert abs(rho - 1.0) < 1e-5
    assert p < 0.01


def test_spearman_perfect_negative():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    rho, p = spearman_correlation(x, y)
    assert abs(rho + 1.0) < 1e-5


def test_cohen_d_equal_groups():
    g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(cohen_d(g1, g2)) < 1e-3


def test_cohen_d_large_effect():
    g1 = np.ones(100) * 10.0
    g2 = np.zeros(100)
    assert cohen_d(g1, g2) > 5.0


def test_report_experiment3_correlation():
    report = report_experiment3_correlation(
        agent_names=["corner_biased", "uniform", "anti_corner"],
        corner_reinforcement_freqs=[0.75, 0.25, 0.10],
        corner_shard_strengths=[0.8, 0.4, 0.1],
    )
    assert report["spearman_rho"] > 0.9
    assert report["meets_target"] is True


def test_report_experiment3_required_keys():
    report = report_experiment3_correlation(["a", "b", "c"], [0.1, 0.5, 0.9], [0.2, 0.5, 0.8])
    assert {"agents", "spearman_rho", "p_value", "significant", "meets_target"}.issubset(
        report.keys()
    )


def test_visualizer_probe_heatmap():
    viz = ShardVisualizer()
    probe_results = {
        "cheese_presence": {
            "block1": {"in_context_acc": 0.7, "out_context_acc": 0.5},
            "fc": {"in_context_acc": 0.95, "out_context_acc": 0.52},
        },
    }
    fig = viz.plot_probe_heatmap(probe_results)
    assert fig is not None
    plt.close("all")


def test_visualizer_causal_effects():
    viz = ShardVisualizer()
    fig = viz.plot_causal_effects({"block1": 0.1, "fc": 0.7})
    assert fig is not None
    plt.close("all")


def test_visualizer_independence_scores():
    viz = ShardVisualizer()
    fig = viz.plot_independence_scores({"I_cheese_corner": 0.85, "I_corner_cheese": 0.90})
    assert fig is not None
    plt.close("all")


def test_visualizer_reinforcement_correlation():
    viz = ShardVisualizer()
    fig = viz.plot_reinforcement_correlation(
        agent_names=["A", "B", "C"],
        corner_freqs=[0.75, 0.25, 0.10],
        corner_strengths=[0.8, 0.4, 0.1],
        rho=0.99,
        p_value=0.01,
    )
    assert fig is not None
    plt.close("all")
