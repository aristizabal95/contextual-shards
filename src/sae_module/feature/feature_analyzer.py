"""Feature analysis for SAE-discovered features (Experiment 4)."""
import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
from scipy.stats import pearsonr
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyzes SAE features to identify those corresponding to shard concepts.

    Maps SAE features to context profiles by computing mean activation contrast
    (concept ON vs OFF) for each feature. High-contrast features correspond to
    concept-encoding directions — these are compared against supervised probes
    for cross-method validation.

    Args:
        sae: trained SparseAutoencoder
    """

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae

    def compute_context_profiles(
        self,
        activations: np.ndarray,
        concept_labels: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Compute mean SAE feature activation contrast per concept.

        For each concept and each SAE feature, computes:
            contrast[feature] = mean(feature | concept ON) - mean(feature | concept OFF)

        High positive contrast -> feature activates more when concept is present.

        Args:
            activations: (N, d_input) raw layer activations (before SAE)
            concept_labels: {concept: (N,) float labels}
            threshold: binarization threshold for concept labels

        Returns:
            {concept: (d_hidden,) contrast vector}
        """
        x = torch.from_numpy(activations).float()
        with torch.no_grad():
            _, features = self.sae(x.view(len(x), -1))
        features_np = features.numpy()

        profiles: Dict[str, np.ndarray] = {}
        for concept, labels in concept_labels.items():
            mask_on = labels > threshold
            mask_off = ~mask_on
            mean_on = features_np[mask_on].mean(0) if mask_on.any() else np.zeros(features_np.shape[1])
            mean_off = features_np[mask_off].mean(0) if mask_off.any() else np.zeros(features_np.shape[1])
            profiles[concept] = mean_on - mean_off
        return profiles

    def top_features_per_concept(
        self,
        profiles: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Return indices of top-k features per concept ranked by |contrast|.

        Args:
            profiles: output of compute_context_profiles()
            top_k: number of top features to return per concept

        Returns:
            {concept: [(feature_idx, contrast_score), ...]} sorted descending by |score|
        """
        result: Dict[str, List[Tuple[int, float]]] = {}
        for concept, profile in profiles.items():
            ranked = sorted(enumerate(profile), key=lambda x: -abs(x[1]))
            result[concept] = [(int(idx), float(score)) for idx, score in ranked[:top_k]]
        return result

    def correlate_feature_with_probe(
        self,
        feature_activations: np.ndarray,
        probe_predictions: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute Pearson correlation between SAE feature and probe predictions.

        Args:
            feature_activations: (N,) activation values for a single SAE feature
            probe_predictions: (N,) predictions from a trained linear probe

        Returns:
            (r, p_value) - Pearson r and two-tailed p-value
            Target per research plan: r > 0.7 for cross-method convergence
        """
        r, p = pearsonr(feature_activations, probe_predictions)
        return float(r), float(p)

    def find_matching_features(
        self,
        activations: np.ndarray,
        probe_predictions: Dict[str, np.ndarray],
        r_threshold: float = 0.7,
    ) -> Dict[str, List[Tuple[int, float, float]]]:
        """Find SAE features that correlate with probe predictions per concept.

        Args:
            activations: (N, d_input) raw activations
            probe_predictions: {concept: (N,) probe prediction scores}
            r_threshold: minimum Pearson |r| to report a feature as matching

        Returns:
            {concept: [(feature_idx, r, p_value), ...]} for features with |r| >= threshold
        """
        x = torch.from_numpy(activations).float()
        with torch.no_grad():
            _, features = self.sae(x.view(len(x), -1))
        features_np = features.numpy()  # (N, d_hidden)

        matches: Dict[str, List[Tuple[int, float, float]]] = {}
        for concept, preds in probe_predictions.items():
            concept_matches = []
            for feat_idx in range(features_np.shape[1]):
                feat_acts = features_np[:, feat_idx]
                if feat_acts.std() < 1e-8:
                    continue  # skip constant features
                r, p = self.correlate_feature_with_probe(feat_acts, preds)
                if abs(r) >= r_threshold:
                    concept_matches.append((feat_idx, r, p))
            concept_matches.sort(key=lambda x: -abs(x[1]))
            matches[concept] = concept_matches
        return matches
