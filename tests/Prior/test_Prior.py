"""Test prior distribution models for MaxLEV."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.prior import (
    UniformPrior,
    GaussianPrior,
    AsymmetricGaussianPrior,
    LogUniformPrior,
    PriorCollection,
    flistCreatePriors,
)


class TestUniformPrior:
    """Tests for UniformPrior."""

    def test_returns_zero(self):
        """Uniform prior contributes zero log-prior."""
        prior = UniformPrior()
        assert prior.fdLogPrior(0.5) == 0.0

    def test_returns_zero_for_any_value(self):
        """Uniform prior returns zero regardless of parameter value."""
        prior = UniformPrior()
        for dValue in [-100.0, 0.0, 1.0, 1e10]:
            assert prior.fdLogPrior(dValue) == 0.0


class TestGaussianPrior:
    """Tests for GaussianPrior."""

    def test_at_mean_returns_zero(self):
        """Gaussian prior at mean returns zero log-prior."""
        prior = GaussianPrior(dMean=1.0, dStd=0.1)
        assert prior.fdLogPrior(1.0) == 0.0

    def test_one_sigma_returns_negative_half(self):
        """Gaussian prior at 1-sigma returns -0.5."""
        prior = GaussianPrior(dMean=1.0, dStd=0.1)
        assert prior.fdLogPrior(1.1) == pytest.approx(-0.5)
        assert prior.fdLogPrior(0.9) == pytest.approx(-0.5)

    def test_two_sigma_returns_negative_two(self):
        """Gaussian prior at 2-sigma returns -2.0."""
        prior = GaussianPrior(dMean=1.0, dStd=0.1)
        assert prior.fdLogPrior(1.2) == pytest.approx(-2.0)

    def test_symmetric(self):
        """Gaussian prior is symmetric around the mean."""
        prior = GaussianPrior(dMean=5.0, dStd=2.0)
        assert prior.fdLogPrior(3.0) == pytest.approx(prior.fdLogPrior(7.0))


class TestAsymmetricGaussianPrior:
    """Tests for AsymmetricGaussianPrior."""

    def test_at_mean_returns_zero(self):
        """Asymmetric Gaussian at mean returns zero."""
        prior = AsymmetricGaussianPrior(dMean=1.0, dStdUpper=0.2, dStdLower=0.1)
        assert prior.fdLogPrior(1.0) == 0.0

    def test_uses_upper_std_above_mean(self):
        """Uses std_upper when value > mean."""
        prior = AsymmetricGaussianPrior(dMean=1.0, dStdUpper=0.2, dStdLower=0.1)
        dExpected = -0.5 * ((1.2 - 1.0) / 0.2) ** 2
        assert prior.fdLogPrior(1.2) == pytest.approx(dExpected)

    def test_uses_lower_std_below_mean(self):
        """Uses std_lower when value < mean."""
        prior = AsymmetricGaussianPrior(dMean=1.0, dStdUpper=0.2, dStdLower=0.1)
        dExpected = -0.5 * ((0.9 - 1.0) / 0.1) ** 2
        assert prior.fdLogPrior(0.9) == pytest.approx(dExpected)

    def test_equal_stds_matches_gaussian(self):
        """Asymmetric with equal stds matches symmetric Gaussian."""
        asymmetric = AsymmetricGaussianPrior(dMean=1.0, dStdUpper=0.5, dStdLower=0.5)
        symmetric = GaussianPrior(dMean=1.0, dStd=0.5)
        for dValue in [0.5, 0.8, 1.0, 1.2, 1.5]:
            assert asymmetric.fdLogPrior(dValue) == pytest.approx(
                symmetric.fdLogPrior(dValue)
            )


class TestLogUniformPrior:
    """Tests for LogUniformPrior."""

    def test_positive_value_returns_negative_log(self):
        """Log-uniform prior returns -ln(x) for positive values."""
        prior = LogUniformPrior()
        assert prior.fdLogPrior(1.0) == pytest.approx(0.0)
        assert prior.fdLogPrior(np.e) == pytest.approx(-1.0)
        assert prior.fdLogPrior(10.0) == pytest.approx(-np.log(10.0))

    def test_zero_returns_negative_infinity(self):
        """Log-uniform prior returns -inf at zero."""
        prior = LogUniformPrior()
        assert prior.fdLogPrior(0.0) == -np.inf

    def test_negative_value_returns_negative_infinity(self):
        """Log-uniform prior returns -inf for negative values."""
        prior = LogUniformPrior()
        assert prior.fdLogPrior(-1.0) == -np.inf

    def test_small_positive_value(self):
        """Log-uniform prior penalizes small values less than large ones."""
        prior = LogUniformPrior()
        assert prior.fdLogPrior(0.01) > prior.fdLogPrior(100.0)

    def test_monotonically_decreasing(self):
        """Log-prior decreases as value increases (favors smaller scales)."""
        prior = LogUniformPrior()
        daValues = [0.1, 1.0, 10.0, 100.0, 1000.0]
        listLogPriors = [prior.fdLogPrior(dVal) for dVal in daValues]
        for i in range(len(listLogPriors) - 1):
            assert listLogPriors[i] > listLogPriors[i + 1]


class TestPriorCollection:
    """Tests for PriorCollection."""

    def test_all_uniform_returns_zero(self):
        """Collection of uniform priors returns zero."""
        collection = PriorCollection([UniformPrior(), UniformPrior()])
        assert collection.fdLogPrior(np.array([1.0, 2.0])) == 0.0

    def test_sums_individual_priors(self):
        """Collection correctly sums log-priors."""
        prior0 = GaussianPrior(dMean=1.0, dStd=0.1)
        prior1 = GaussianPrior(dMean=2.0, dStd=0.5)
        collection = PriorCollection([prior0, prior1])
        daTheta = np.array([1.1, 2.5])
        dExpected = prior0.fdLogPrior(1.1) + prior1.fdLogPrior(2.5)
        assert collection.fdLogPrior(daTheta) == pytest.approx(dExpected)

    def test_neg_log_prior_negates(self):
        """fdNegLogPrior returns negative of fdLogPrior."""
        prior = GaussianPrior(dMean=1.0, dStd=0.1)
        collection = PriorCollection([prior])
        daTheta = np.array([1.2])
        assert collection.fdNegLogPrior(daTheta) == pytest.approx(
            -collection.fdLogPrior(daTheta)
        )

    def test_has_priors_false_for_all_uniform(self):
        """fbHasPriors returns False when all priors are uniform."""
        collection = PriorCollection([UniformPrior(), UniformPrior()])
        assert collection.fbHasPriors() is False

    def test_has_priors_true_with_gaussian(self):
        """fbHasPriors returns True when any Gaussian prior exists."""
        collection = PriorCollection([
            UniformPrior(), GaussianPrior(dMean=1.0, dStd=0.1)
        ])
        assert collection.fbHasPriors() is True


class TestCreatePriors:
    """Tests for flistCreatePriors factory function."""

    def test_creates_uniform_by_default(self):
        """Factory creates uniform prior when type is 'uniform'."""
        collection = flistCreatePriors([{"type": "uniform"}])
        assert isinstance(collection.listPriors[0], UniformPrior)

    def test_creates_gaussian(self):
        """Factory creates Gaussian prior from config dict."""
        collection = flistCreatePriors([
            {"type": "gaussian", "mean": 1.0, "std": 0.1}
        ])
        assert isinstance(collection.listPriors[0], GaussianPrior)
        assert collection.listPriors[0].dMean == 1.0
        assert collection.listPriors[0].dStd == 0.1

    def test_creates_asymmetric_gaussian(self):
        """Factory creates asymmetric Gaussian from config dict."""
        collection = flistCreatePriors([{
            "type": "asymmetric_gaussian",
            "mean": 1.0,
            "std_upper": 0.2,
            "std_lower": 0.1,
        }])
        assert isinstance(collection.listPriors[0], AsymmetricGaussianPrior)

    def test_creates_log_uniform(self):
        """Factory creates log-uniform prior from config dict."""
        collection = flistCreatePriors([{"type": "log_uniform"}])
        assert isinstance(collection.listPriors[0], LogUniformPrior)

    def test_unknown_type_raises(self):
        """Factory raises ValueError for unknown prior type."""
        with pytest.raises(ValueError, match="Unknown prior type"):
            flistCreatePriors([{"type": "laplacian"}])

    def test_mixed_priors(self):
        """Factory correctly handles mix of uniform and Gaussian."""
        collection = flistCreatePriors([
            {"type": "gaussian", "mean": 1.0, "std": 0.1},
            {"type": "uniform"},
            {"type": "asymmetric_gaussian", "mean": 2.0, "std_upper": 0.3, "std_lower": 0.2},
        ])
        assert len(collection.listPriors) == 3
        assert isinstance(collection.listPriors[0], GaussianPrior)
        assert isinstance(collection.listPriors[1], UniformPrior)
        assert isinstance(collection.listPriors[2], AsymmetricGaussianPrior)
        assert collection.fbHasPriors() is True

    def test_default_type_is_uniform(self):
        """Factory defaults to uniform when no type specified."""
        collection = flistCreatePriors([{}])
        assert isinstance(collection.listPriors[0], UniformPrior)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
