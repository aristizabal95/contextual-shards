import os

import numpy as np
from typing import Dict, List
from src.probe_module import ProbeFactory


def _ram_mb() -> str:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024 ** 2
        avail_mb = psutil.virtual_memory().available / 1024 ** 2
        return f"RSS={rss_mb:.0f} MB  avail={avail_mb:.0f} MB"
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
            return f"avail={lines.get('MemAvailable', 0) // 1024} MB"
        except Exception:
            return "(RAM unavailable)"


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
        print(f"    [EVAL] X shape={X.shape} n_in={context_mask.sum()} n_out={(~context_mask).sum()} — {_ram_mb()}")

        in_accs: List[float] = []
        out_accs: List[float] = []

        for trial_i in range(n_trials):
            # Train on a random 80% split of ALL data so both classes are always present.
            # context_mask is derived from the same signal as y, so X_in contains only
            # positive labels and X_out only negative labels — training on either subset
            # alone produces a single-class training set.
            print(f"    [EVAL] trial {trial_i + 1}/{n_trials} fitting probe — {_ram_mb()}")
            idx = np.random.permutation(len(X))
            split = max(4, int(0.8 * len(idx)))
            train_idx, test_idx = idx[:split], idx[split:]

            if len(np.unique(y[train_idx])) < 2:
                print(f"    [EVAL] trial {trial_i + 1}/{n_trials} skipped: single class in train split")
                continue

            probe = ProbeFactory(self.probe_name)()
            probe.fit(X[train_idx], y[train_idx])
            print(f"    [EVAL] trial {trial_i + 1}/{n_trials} scoring — {_ram_mb()}")

            # Evaluate separately on in-context and out-of-context TEST samples
            in_test = test_idx[context_mask[test_idx]]
            out_test = test_idx[~context_mask[test_idx]]

            if len(in_test) >= 2:
                in_accs.append(probe.score(X[in_test], y[in_test]))
            if len(out_test) >= 2:
                out_accs.append(probe.score(X[out_test], y[out_test]))
            print(f"    [EVAL] trial {trial_i + 1}/{n_trials} done — {_ram_mb()}")

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
