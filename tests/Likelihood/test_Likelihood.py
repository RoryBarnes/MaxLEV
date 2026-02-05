"""Test likelihood models for MaxLEV."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.likelihood import (
    GaussianLikelihood,
    AsymmetricGaussianLikelihood,
    create_likelihood,
)
from maxlev.config import ObservableConfig


class TestGaussianLikelihood:
    """Tests for GaussianLikelihood class."""

    def test_perfect_fit_returns_zero(self):
        """Test that perfect fit returns zero neg_log_likelihood."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        likelihood = GaussianLikelihood(return_negative=True)
        computed = {'test': 1.0}

        result = likelihood.compute(computed, [obs])
        assert result == 0.0

    def test_one_sigma_deviation(self):
        """Test that 1-sigma deviation returns 0.5 neg_log_likelihood."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        likelihood = GaussianLikelihood(return_negative=True)
        computed = {'test': 1.1}  # 1 sigma away

        result = likelihood.compute(computed, [obs])
        assert abs(result - 0.5) < 1e-10

    def test_two_sigma_deviation(self):
        """Test that 2-sigma deviation returns 2.0 neg_log_likelihood."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        likelihood = GaussianLikelihood(return_negative=True)
        computed = {'test': 1.2}  # 2 sigma away

        result = likelihood.compute(computed, [obs])
        assert abs(result - 2.0) < 1e-10

    def test_multiple_observables(self):
        """Test likelihood with multiple observables."""
        obs1 = ObservableConfig(
            name='obs1',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        obs2 = ObservableConfig(
            name='obs2',
            type='direct',
            observed_value=2.0,
            uncertainty=0.2
        )

        likelihood = GaussianLikelihood(return_negative=True)
        computed = {'obs1': 1.1, 'obs2': 2.2}  # both 1 sigma

        result = likelihood.compute(computed, [obs1, obs2])
        # chi2 = 1^2 + 1^2 = 2, neg_log_like = 0.5 * 2 = 1.0
        assert abs(result - 1.0) < 1e-10

    def test_return_positive_likelihood(self):
        """Test returning positive log likelihood."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        likelihood = GaussianLikelihood(return_negative=False)
        computed = {'test': 1.1}

        result = likelihood.compute(computed, [obs])
        # log(L) = -0.5 * chi2 = -0.5
        assert abs(result - (-0.5)) < 1e-10


class TestAsymmetricGaussianLikelihood:
    """Tests for AsymmetricGaussianLikelihood class."""

    def test_uses_lower_uncertainty_below(self):
        """Test that lower uncertainty is used when computed < observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.5
        )

        likelihood = AsymmetricGaussianLikelihood(return_negative=True)
        computed = {'test': 0.9}  # 1 sigma below using sigma_lower

        result = likelihood.compute(computed, [obs])
        # chi2 = ((0.9 - 1.0) / 0.1)^2 = 1.0
        assert abs(result - 0.5) < 1e-10

    def test_uses_upper_uncertainty_above(self):
        """Test that upper uncertainty is used when computed >= observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.5
        )

        likelihood = AsymmetricGaussianLikelihood(return_negative=True)
        computed = {'test': 1.5}  # 1 sigma above using sigma_upper

        result = likelihood.compute(computed, [obs])
        # chi2 = ((1.5 - 1.0) / 0.5)^2 = 1.0
        assert abs(result - 0.5) < 1e-10

    def test_asymmetric_vs_symmetric_same_at_center(self):
        """Test asymmetric equals symmetric when uncertainties are equal."""
        obs_sym = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.2
        )
        obs_asym = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.2,
            uncertainty_upper=0.2
        )

        like_sym = GaussianLikelihood(return_negative=True)
        like_asym = AsymmetricGaussianLikelihood(return_negative=True)

        computed = {'test': 1.0}

        result_sym = like_sym.compute(computed, [obs_sym])
        result_asym = like_asym.compute(computed, [obs_asym])

        assert abs(result_sym - result_asym) < 1e-10


class TestCreateLikelihood:
    """Tests for create_likelihood factory function."""

    def test_creates_gaussian_for_symmetric(self):
        """Test that Gaussian is created for symmetric uncertainties."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        result = create_likelihood({}, [obs])

        assert isinstance(result, GaussianLikelihood)

    def test_creates_asymmetric_for_asymmetric(self):
        """Test that Asymmetric is created when any observable is asymmetric."""
        obs_sym = ObservableConfig(
            name='obs1',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        obs_asym = ObservableConfig(
            name='obs2',
            type='direct',
            observed_value=2.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )

        result = create_likelihood({}, [obs_sym, obs_asym])

        assert isinstance(result, AsymmetricGaussianLikelihood)

    def test_respects_return_negative_config(self):
        """Test that return_negative is passed from config."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        result = create_likelihood({'return_negative': False}, [obs])

        assert result.return_negative is False

    def test_default_return_negative_true(self):
        """Test that default return_negative is True."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )

        result = create_likelihood({}, [obs])

        assert result.return_negative is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
