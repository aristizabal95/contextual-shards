import pytest
from src.trainer_module.rl_trainer.cheese_distribution import CheesePlacementDistribution


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        CheesePlacementDistribution(mode="invalid")


def test_uniform_returns_valid_positions():
    dist = CheesePlacementDistribution("uniform", grid_size=15, seed=42)
    for _ in range(100):
        r, c = dist.sample()
        assert 0 <= r < 15
        assert 0 <= c < 15


def test_corner_biased_skews_to_top_right():
    dist = CheesePlacementDistribution("corner_biased", grid_size=15, seed=0)
    fraction = dist.empirical_corner_fraction(n_samples=5000)
    # Should be ~75% * (1 for corner) + 25% * (0.25 for corner) = ~63.75% roughly
    # More precisely: 75% in corner, 25% uniform (25% of that in corner = 6.25%)
    # Total ≈ 75 + 6.25 = ~81% but corner quadrant is half*half / full = 1/4 for uniform part
    # Actual: 75% guaranteed in corner + 25% * 0.25 ≈ 81.25%
    assert fraction > 0.6, f"Expected > 0.6 corner fraction, got {fraction:.3f}"


def test_uniform_roughly_equal_quadrants():
    dist = CheesePlacementDistribution("uniform", grid_size=15, seed=1)
    fraction = dist.empirical_corner_fraction(n_samples=5000)
    # Top-right quadrant: rows [0,7), cols [7,15) = 7*8 = 56 / 225 total ≈ 0.25
    assert 0.15 < fraction < 0.40, f"Expected ~0.25 corner fraction, got {fraction:.3f}"


def test_anti_corner_avoids_top_right():
    dist = CheesePlacementDistribution("anti_corner", grid_size=15, seed=2)
    fraction = dist.empirical_corner_fraction(n_samples=5000)
    # 10% in corner + 20% uniform * 0.25 = ~15%
    assert fraction < 0.25, f"Expected < 0.25 corner fraction, got {fraction:.3f}"


def test_corner_biased_higher_than_anti_corner():
    dist_biased = CheesePlacementDistribution("corner_biased", grid_size=15, seed=99)
    dist_anti = CheesePlacementDistribution("anti_corner", grid_size=15, seed=99)
    f_biased = dist_biased.empirical_corner_fraction(n_samples=2000)
    f_anti = dist_anti.empirical_corner_fraction(n_samples=2000)
    assert f_biased > f_anti


def test_reproducibility_with_seed():
    dist1 = CheesePlacementDistribution("uniform", grid_size=15, seed=42)
    dist2 = CheesePlacementDistribution("uniform", grid_size=15, seed=42)
    samples1 = [dist1.sample() for _ in range(10)]
    samples2 = [dist2.sample() for _ in range(10)]
    assert samples1 == samples2


def test_all_modes_return_tuples():
    for mode in ("corner_biased", "uniform", "anti_corner"):
        dist = CheesePlacementDistribution(mode, grid_size=15, seed=0)
        r, c = dist.sample()
        assert isinstance(r, int)
        assert isinstance(c, int)
