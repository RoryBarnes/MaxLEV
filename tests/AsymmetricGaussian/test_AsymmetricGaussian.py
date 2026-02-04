"""Test asymmetric Gaussian likelihood normalization."""

import pytest
import numpy as np
from scipy import integrate
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.likelihood import AsymmetricGaussianLikelihood, GaussianLikelihood
from maxlev.config import ObservableConfig


def asymmetric_gaussian_pdf(x, mu, sigma_lower, sigma_upper):
    """
    Asymmetric Gaussian PDF (not normalized).

    Uses sigma_lower when x < mu, sigma_upper when x >= mu.
    """
    sigma = sigma_lower if x < mu else sigma_upper
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_asymmetric_gaussian_integral():
    """
    Test that the integral under an asymmetric Gaussian equals 1.

    The normalization constant for an asymmetric Gaussian is:
    C = sqrt(2*pi) * (sigma_lower + sigma_upper) / 2

    So the normalized PDF is:
    p(x) = (1/C) * exp(-0.5 * ((x-mu)/sigma)^2)

    The integral from -inf to +inf should equal 1.
    """

    # Test case 1: Moderately asymmetric
    mu = 1.0
    sigma_lower = 0.1
    sigma_upper = 0.3

    # Compute normalization constant
    C = np.sqrt(2 * np.pi) * (sigma_lower + sigma_upper) / 2

    # Define normalized PDF
    def normalized_pdf(x):
        return asymmetric_gaussian_pdf(x, mu, sigma_lower, sigma_upper) / C

    # Integrate from -inf to +inf
    integral, error = integrate.quad(normalized_pdf, mu - 10*sigma_lower, mu + 10*sigma_upper)

    print(f"\nTest case 1: mu={mu}, sigma_lower={sigma_lower}, sigma_upper={sigma_upper}")
    print(f"  Normalization constant C = {C:.6f}")
    print(f"  Integral = {integral:.10f}")
    print(f"  Error = {error:.2e}")

    epsilon = 1e-6
    assert abs(integral - 1.0) < epsilon, f"Integral should be 1, got {integral}"
    print(f"  ✓ |integral - 1| = {abs(integral - 1.0):.2e} < {epsilon}")


def test_asymmetric_gaussian_integral_extreme():
    """Test normalization with very asymmetric sigma values."""

    mu = 0.0
    sigma_lower = 0.01
    sigma_upper = 1.0  # 100x larger!

    C = np.sqrt(2 * np.pi) * (sigma_lower + sigma_upper) / 2

    def normalized_pdf(x):
        return asymmetric_gaussian_pdf(x, mu, sigma_lower, sigma_upper) / C

    integral, error = integrate.quad(normalized_pdf, mu - 10*sigma_lower, mu + 10*sigma_upper)

    print(f"\nTest case 2 (extreme): mu={mu}, sigma_lower={sigma_lower}, sigma_upper={sigma_upper}")
    print(f"  Normalization constant C = {C:.6f}")
    print(f"  Integral = {integral:.10f}")
    print(f"  Error = {error:.2e}")

    epsilon = 1e-6
    assert abs(integral - 1.0) < epsilon, f"Integral should be 1, got {integral}"
    print(f"  ✓ |integral - 1| = {abs(integral - 1.0):.2e} < {epsilon}")


def test_asymmetric_gaussian_reduces_to_symmetric():
    """Test that asymmetric Gaussian reduces to symmetric when sigma_lower = sigma_upper."""

    mu = 5.0
    sigma = 0.5

    C_symmetric = np.sqrt(2 * np.pi) * sigma
    C_asymmetric = np.sqrt(2 * np.pi) * (sigma + sigma) / 2

    # Should be equal
    assert abs(C_symmetric - C_asymmetric) < 1e-10

    def normalized_pdf_sym(x):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / C_symmetric

    def normalized_pdf_asym(x):
        return asymmetric_gaussian_pdf(x, mu, sigma, sigma) / C_asymmetric

    # Sample at several points
    x_values = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

    for x in x_values:
        p_sym = normalized_pdf_sym(x)
        p_asym = normalized_pdf_asym(x)
        assert abs(p_sym - p_asym) < 1e-10, f"At x={x}, symmetric and asymmetric PDFs differ"

    # Check integral
    integral, _ = integrate.quad(normalized_pdf_asym, mu - 10*sigma, mu + 10*sigma)

    print(f"\nTest case 3 (symmetric limit): mu={mu}, sigma={sigma}")
    print(f"  Integral = {integral:.10f}")

    epsilon = 1e-6
    assert abs(integral - 1.0) < epsilon
    print(f"  ✓ Symmetric case: |integral - 1| = {abs(integral - 1.0):.2e} < {epsilon}")


def test_likelihood_computation_asymmetric():
    """
    Test that the AsymmetricGaussianLikelihood class computes correct chi-squared.

    For a Gaussian likelihood:
    -2*ln(L) = chi^2 = sum((x - mu)^2 / sigma^2)

    The normalization constant doesn't affect the chi^2.
    """

    # Create observables with asymmetric uncertainties
    obs1 = ObservableConfig(
        name="obs1",
        type="direct",
        observed_value=1.0,
        uncertainty_lower=0.1,
        uncertainty_upper=0.3
    )

    obs2 = ObservableConfig(
        name="obs2",
        type="direct",
        observed_value=2.0,
        uncertainty_lower=0.2,
        uncertainty_upper=0.2  # Symmetric
    )

    likelihood = AsymmetricGaussianLikelihood(return_negative=True)

    # Test case 1: computed < observed (use sigma_lower)
    computed = {"obs1": 0.9, "obs2": 2.0}

    neg_log_like = likelihood.compute(computed, [obs1, obs2])

    # Manual calculation
    # obs1: (0.9 - 1.0) / 0.1 = -1.0, chi2_contrib = 1.0
    # obs2: (2.0 - 2.0) / 0.2 = 0.0, chi2_contrib = 0.0
    # chi2 = 1.0, -log(L) = 0.5 * chi2 = 0.5

    expected = 0.5
    assert abs(neg_log_like - expected) < 1e-10, f"Expected {expected}, got {neg_log_like}"
    print(f"✓ Asymmetric likelihood (computed < observed): -log(L) = {neg_log_like:.6f}")

    # Test case 2: computed > observed (use sigma_upper)
    computed = {"obs1": 1.2, "obs2": 2.0}

    neg_log_like = likelihood.compute(computed, [obs1, obs2])

    # Manual calculation
    # obs1: (1.2 - 1.0) / 0.3 = 0.667, chi2_contrib = 0.444
    # obs2: (2.0 - 2.0) / 0.2 = 0.0, chi2_contrib = 0.0
    # chi2 = 0.444, -log(L) = 0.5 * chi2 = 0.222

    expected = 0.5 * ((1.2 - 1.0) / 0.3) ** 2
    assert abs(neg_log_like - expected) < 1e-10, f"Expected {expected}, got {neg_log_like}"
    print(f"✓ Asymmetric likelihood (computed > observed): -log(L) = {neg_log_like:.6f}")


def test_symmetric_vs_asymmetric_likelihood():
    """Compare symmetric and asymmetric likelihoods when sigma_lower = sigma_upper."""

    obs = ObservableConfig(
        name="test",
        type="direct",
        observed_value=1.0,
        uncertainty=0.2  # Symmetric
    )

    obs_asym = ObservableConfig(
        name="test",
        type="direct",
        observed_value=1.0,
        uncertainty_lower=0.2,
        uncertainty_upper=0.2  # Asymmetric but equal
    )

    likelihood_sym = GaussianLikelihood(return_negative=True)
    likelihood_asym = AsymmetricGaussianLikelihood(return_negative=True)

    # Test at several computed values
    computed_values = [0.5, 0.8, 1.0, 1.2, 1.5]

    for val in computed_values:
        computed = {"test": val}

        nll_sym = likelihood_sym.compute(computed, [obs])
        nll_asym = likelihood_asym.compute(computed, [obs_asym])

        assert abs(nll_sym - nll_asym) < 1e-10, \
            f"At computed={val}, symmetric and asymmetric differ: {nll_sym} vs {nll_asym}"

    print("✓ Symmetric and asymmetric likelihoods agree when sigma_lower = sigma_upper")


def test_numerical_integration_multiple_cases():
    """Run normalization test for multiple parameter combinations."""

    test_cases = [
        (0.0, 0.1, 0.1),      # Symmetric
        (1.0, 0.05, 0.15),    # Moderately asymmetric
        (5.0, 0.01, 1.0),     # Very asymmetric
        (-2.0, 0.3, 0.1),     # Reverse asymmetry
        (10.0, 1.0, 10.0),    # Large values
    ]

    epsilon = 1e-6

    print(f"\n{'='*70}")
    print("Testing normalization for multiple parameter combinations:")
    print(f"{'='*70}")

    for i, (mu, sigma_l, sigma_u) in enumerate(test_cases, 1):
        C = np.sqrt(2 * np.pi) * (sigma_l + sigma_u) / 2

        def normalized_pdf(x):
            return asymmetric_gaussian_pdf(x, mu, sigma_l, sigma_u) / C

        # Integrate over a reasonable range
        x_min = mu - max(10 * sigma_l, 10)
        x_max = mu + max(10 * sigma_u, 10)
        integral, error = integrate.quad(normalized_pdf, x_min, x_max)

        deviation = abs(integral - 1.0)
        status = "✓" if deviation < epsilon else "✗"

        print(f"{status} Case {i}: mu={mu:6.2f}, σ_l={sigma_l:6.3f}, σ_u={sigma_u:6.3f} "
              f"→ integral={integral:.10f} (Δ={deviation:.2e})")

        assert deviation < epsilon, f"Case {i} failed normalization"

    print(f"{'='*70}")
    print(f"All {len(test_cases)} test cases passed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
