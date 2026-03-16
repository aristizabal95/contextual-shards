"""Publication-ready visualizations for shard analysis experiments."""
import logging
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ShardVisualizer:
    """Generates figures for shard analysis: probe heatmaps, causal effects, independence scores."""

    def plot_probe_heatmap(
        self,
        probe_results: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "in_context_acc",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Heatmap: rows=concepts, cols=layers, color=probe accuracy.

        Args:
            probe_results: {concept: {layer: {metric: value}}}
            metric: which metric to plot (default: in_context_acc)
            output_path: save figure to this path if provided

        Returns:
            matplotlib Figure
        """
        concepts = list(probe_results)
        layers = list(next(iter(probe_results.values())))
        data = np.array(
            [
                [
                    probe_results[c].get(layer, {}).get(metric, float("nan"))
                    for layer in layers
                ]
                for c in concepts
            ]
        )

        fig, ax = plt.subplots(
            figsize=(max(8, len(layers) * 0.8), max(3, len(concepts) * 0.8))
        )
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(concepts)))
        ax.set_yticklabels(concepts, fontsize=9)
        plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
        ax.set_title(f"Probe {metric.replace('_', ' ').title()} by Layer and Concept (D1)")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved probe heatmap to {output_path}")

        return fig

    def plot_causal_effects(
        self,
        causal_effects: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar plot of causal effect size per layer (D2).

        Args:
            causal_effects: {layer: effect_score}
            output_path: optional save path
        """
        layers = list(causal_effects)
        values = [causal_effects[layer] for layer in layers]

        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.8), 4))
        ax.bar(range(len(layers)), values, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.3, color="red", linestyle="--", linewidth=1.5, label="Target >= 0.3")
        ax.set_ylabel("Causal Effect Score")
        ax.set_title("Causal Tracing: Layer-wise Behavioral Recovery (D2)")
        ax.legend()
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved causal effects plot to {output_path}")

        return fig

    def plot_independence_scores(
        self,
        results: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar plot of shard independence scores (D3).

        Args:
            results: dict with keys starting with "I_" (e.g., I_cheese_corner)
            output_path: optional save path
        """
        keys = [k for k in results if k.startswith("I_")]
        values = [results[k] for k in keys]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][: len(keys)]

        fig, ax = plt.subplots(figsize=(max(5, len(keys) * 1.5), 4))
        ax.bar(keys, values, color=colors, alpha=0.85)
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1.5, label="Target I >= 0.8")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Independence Score I")
        ax.set_title("Shard Separability (D3): Independence Scores")
        ax.legend()
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved independence scores plot to {output_path}")

        return fig

    def plot_reinforcement_correlation(
        self,
        agent_names: List[str],
        corner_freqs: List[float],
        corner_strengths: List[float],
        rho: float,
        p_value: float = float("nan"),
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter: reinforcement frequency vs shard strength (D4).

        Args:
            agent_names: labels for each point
            corner_freqs: x-axis (training corner frequency)
            corner_strengths: y-axis (shard strength)
            rho: Spearman correlation coefficient
            p_value: p-value for the correlation
            output_path: optional save path
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(corner_freqs, corner_strengths, s=120, color="darkorange", zorder=3)
        for name, x, y in zip(agent_names, corner_freqs, corner_strengths):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
        ax.set_xlabel("Corner Reinforcement Frequency (training)")
        ax.set_ylabel("Corner Shard Strength")
        p_str = f", p={p_value:.3f}" if not np.isnan(p_value) else ""
        ax.set_title(f"Reinforcement Traceability (D4) -- Spearman rho={rho:.2f}{p_str}")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved reinforcement correlation to {output_path}")

        return fig
