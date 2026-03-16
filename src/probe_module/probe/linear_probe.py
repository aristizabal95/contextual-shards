import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.probe_module.base_probe import BaseProbe
from src.probe_module import register_probe


@register_probe("linear")
class LinearProbe(BaseProbe):
    """Logistic regression probe over flattened activations."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = self._scaler.fit_transform(self._flatten(X))
        self._clf.fit(Xf, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(self._flatten(X))
        return self._clf.predict(Xf)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        Xf = self._scaler.transform(self._flatten(X))
        return float(self._clf.score(Xf, y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = self._scaler.transform(self._flatten(X))
        return self._clf.predict_proba(Xf)
