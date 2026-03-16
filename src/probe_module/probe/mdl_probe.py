"""MDL probing via online-coding codelength (Voita & Titov, 2020 - simplified)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe_module.base_probe import BaseProbe
from src.probe_module import register_probe


@register_probe("mdl")
class MDLProbe(BaseProbe):
    """MDL probe: reports minimum description length (codelength) in addition to accuracy.

    Codelength is computed via online coding: train on progressively larger fractions
    and sum up the log-loss on each new chunk.
    """

    def __init__(self, C: float = 1.0):
        self._clf = LogisticRegression(C=C, max_iter=1000)
        self._scaler = StandardScaler()
        self._codelength: float = float("inf")
        self._fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        Xf = self._scaler.fit_transform(X.reshape(n, -1))
        n_classes = len(np.unique(y))
        codelength = 0.0

        prev_end = 0
        for i, frac in enumerate(self._fractions):
            end = int(frac * n)
            if end <= prev_end:
                continue
            start = prev_end

            if i == 0:
                # Uniform prior: log2(n_classes) bits per sample
                codelength += (end - start) * np.log2(max(n_classes, 2))
            else:
                # Use current model to encode this chunk
                proba = self._clf.predict_proba(Xf[start:end])
                labels = y[start:end].astype(int)
                # Map labels to class indices
                classes = list(self._clf.classes_)
                eps = 1e-10
                codelength += -sum(
                    np.log2(proba[j, classes.index(labels[j])] + eps)
                    for j in range(len(labels))
                )

            self._clf.fit(Xf[:end], y[:end])
            prev_end = end

        self._codelength = codelength

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(X.reshape(len(X), -1))
        return self._clf.predict(Xf)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        Xf = self._scaler.transform(X.reshape(len(X), -1))
        return float(self._clf.score(Xf, y))

    @property
    def codelength(self) -> float:
        """Lower codelength = probe learned a more compact representation."""
        return self._codelength
