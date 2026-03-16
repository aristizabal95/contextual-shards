import torch
import pytest
from src.sae_module.model.sparse_autoencoder import SparseAutoencoder
from src.sae_module.training.sae_trainer import SAETrainer
from src.sae_module.feature.feature_analyzer import FeatureAnalyzer
import numpy as np


class TestSparseAutoencoder:
    def test_forward_output_shapes(self):
        sae = SparseAutoencoder(d_input=64, expansion_factor=4)
        x = torch.randn(16, 64)
        recon, features = sae(x)
        assert recon.shape == (16, 64)
        assert features.shape == (16, 64 * 4)

    def test_features_nonnegative(self):
        """Features must be non-negative (ReLU activation)."""
        sae = SparseAutoencoder(d_input=32, expansion_factor=8)
        x = torch.randn(50, 32)
        _, features = sae(x)
        assert (features >= 0).all()

    def test_loss_finite(self):
        sae = SparseAutoencoder(d_input=16, expansion_factor=4)
        x = torch.randn(8, 16)
        recon, features = sae(x)
        loss = sae.loss(x, recon, features)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_training(self):
        """SAE loss should decrease over several gradient steps."""
        torch.manual_seed(42)
        sae = SparseAutoencoder(d_input=32, expansion_factor=4, l1_coef=0.01)
        x = torch.randn(64, 32)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        initial_loss = None
        for step in range(20):
            optimizer.zero_grad()
            recon, features = sae(x)
            loss = sae.loss(x, recon, features)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

        final_loss = loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_normalize_decoder_unit_norm(self):
        sae = SparseAutoencoder(d_input=16, expansion_factor=4)
        # Manually corrupt decoder norms
        with torch.no_grad():
            sae.W_dec.weight.data *= 5.0
        sae.normalize_decoder()
        col_norms = sae.W_dec.weight.data.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    def test_factory_registration(self):
        from src.sae_module import SAEFactory
        cls = SAEFactory("standard")
        assert cls is SparseAutoencoder

    def test_encode_decode_roundtrip_after_training(self):
        """After some training, reconstruction should be closer to input."""
        torch.manual_seed(0)
        sae = SparseAutoencoder(d_input=16, expansion_factor=4, l1_coef=0.001)
        x = torch.randn(32, 16)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(50):
            optimizer.zero_grad()
            recon, features = sae(x)
            loss = sae.loss(x, recon, features)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
        recon, _ = sae(x)
        mse = ((x - recon) ** 2).mean().item()
        assert mse < 2.0  # some reconstruction quality


class TestSAETrainer:
    def test_train_on_tensor_returns_losses(self):
        sae = SparseAutoencoder(d_input=16, expansion_factor=4)
        trainer = SAETrainer(sae, lr=1e-3, n_epochs=3, batch_size=8)
        X = torch.randn(32, 16)
        losses = trainer.train_on_tensor(X)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_train_on_tensor_loss_finite(self):
        sae = SparseAutoencoder(d_input=8, expansion_factor=2)
        trainer = SAETrainer(sae, n_epochs=2, batch_size=4)
        X = torch.randn(10, 8)
        losses = trainer.train_on_tensor(X)
        assert all(np.isfinite(l) for l in losses)

    def test_save_and_load(self, tmp_path):
        sae = SparseAutoencoder(d_input=16, expansion_factor=2)
        trainer = SAETrainer(sae)
        path = str(tmp_path / "sae.pt")
        trainer.save(path)
        # Load into new instance
        sae2 = SparseAutoencoder(d_input=16, expansion_factor=2)
        trainer2 = SAETrainer(sae2)
        trainer2.load(path)
        # Weights should match
        for p1, p2 in zip(sae.parameters(), sae2.parameters()):
            assert torch.allclose(p1, p2)


class TestFeatureAnalyzer:
    def _make_trained_sae(self, d_input=32, expansion=4):
        sae = SparseAutoencoder(d_input=d_input, expansion_factor=expansion)
        return sae

    def test_compute_context_profiles_shapes(self):
        sae = self._make_trained_sae(d_input=32, expansion=4)
        analyzer = FeatureAnalyzer(sae)
        acts = np.random.randn(100, 32).astype(np.float32)
        labels = {
            "cheese_presence": (np.random.rand(100) > 0.5).astype(np.float32),
        }
        profiles = analyzer.compute_context_profiles(acts, labels)
        assert "cheese_presence" in profiles
        assert profiles["cheese_presence"].shape == (32 * 4,)

    def test_top_features_per_concept(self):
        sae = self._make_trained_sae(d_input=32, expansion=4)
        analyzer = FeatureAnalyzer(sae)
        profiles = {"cheese": np.random.randn(128)}
        top = analyzer.top_features_per_concept(profiles, top_k=5)
        assert len(top["cheese"]) == 5
        scores = [abs(s) for _, s in top["cheese"]]
        assert scores == sorted(scores, reverse=True)

    def test_correlate_feature_with_probe(self):
        sae = self._make_trained_sae()
        analyzer = FeatureAnalyzer(sae)
        feat = np.random.rand(100)
        probe = feat + np.random.randn(100) * 0.1  # highly correlated
        r, p = analyzer.correlate_feature_with_probe(feat, probe)
        assert r > 0.9
        assert p < 0.01
