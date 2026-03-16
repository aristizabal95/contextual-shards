import numpy as np
from typing import Dict, List
from src.probe_module import ProbeFactory


class ProbeEvaluator:
    """Evaluates probes across contexts with statistical testing.

    Supports layer-wise evaluation and context-split accuracy measurement
    for detecting contextual encoding (Shard Theory criterion D1).
    """

    def __init__(self, probe_name: str = "linear", n_trials: int = 5):
        self.probe_name = probe_name
        self.n_trials = n_trials

    def evaluate_context_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_mask: np.ndarray,
        n_trials: int = 0,
    ) -> Dict[str, float]:
        """Train probe on in-context samples, report accuracy in vs out of context.

        Args:
            X: activations (N, ...)
            y: binary labels (N,)
            context_mask: boolean mask, True = concept context active (N,)
            n_trials: number of random train/test splits (0 = use self.n_trials)

        Returns:
            dict with in_context_acc, in_context_std, out_context_acc
        """
        n_trials = n_trials or self.n_trials
        X_in, y_in = X[context_mask], y[context_mask]
        X_out, y_out = X[~context_mask], y[~context_mask]

        in_accs: List[float] = []
        out_accs: List[float] = []

        for _ in range(n_trials):
            if len(X_in) < 4:
                break
            idx = np.random.permutation(len(X_in))
            split = max(2, int(0.8 * len(idx)))
            probe = ProbeFactory(self.probe_name)()
            probe.fit(X_in[idx[:split]], y_in[idx[:split]])
            in_accs.append(probe.score(X_in[idx[split:]], y_in[idx[split:]]))
            if len(X_out) >= 2:
                out_accs.append(probe.score(X_out, y_out))

        return {
            "in_context_acc": float(np.mean(in_accs)) if in_accs else float("nan"),
            "in_context_std": float(np.std(in_accs)) if in_accs else float("nan"),
            "out_context_acc": float(np.mean(out_accs)) if out_accs else float("nan"),
        }

    def evaluate_all_layers(
        self,
        activations_by_layer: Dict[str, np.ndarray],
        y: np.ndarray,
        context_mask: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate probe at each layer. Returns {layer: {metric: value}}."""
        results = {}
        for layer, X in activations_by_layer.items():
            results[layer] = self.evaluate_context_split(X, y, context_mask)
        return results

    def bonferroni_threshold(self, n_comparisons: int, alpha: float = 0.01) -> float:
        """Return Bonferroni-corrected significance threshold."""
        return alpha / max(n_comparisons, 1)
