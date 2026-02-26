"""Likelihood models for MaxLEV."""

import numpy as np
from typing import Dict
from abc import ABC, abstractmethod


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models."""

    @abstractmethod
    def compute(self, computed_values: Dict[str, float], observables: list) -> float:
        """Compute log-likelihood given computed and observed values."""
        pass


class GaussianLikelihood(LikelihoodModel):
    """Gaussian (chi-squared) likelihood model."""

    def __init__(self, return_negative: bool = True):
        """
        Initialize Gaussian likelihood.

        Args:
            return_negative: If True, return -log(L) for minimization
        """
        self.return_negative = return_negative

    def compute(self, computed_values: Dict[str, float], observables: list) -> float:
        """
        Compute Gaussian log-likelihood.

        Args:
            computed_values: Dict mapping observable name to computed value
            observables: List of ObservableConfig with observed values

        Returns:
            Log-likelihood (or negative log-likelihood if return_negative=True)
        """
        chi2 = 0.0

        for obs in observables:
            computed = computed_values[obs.name]
            observed = obs.observed_value
            sigma = obs.fdGetUncertainty(computed)

            chi2 += ((computed - observed) / sigma) ** 2

        log_likelihood = -0.5 * chi2

        if self.return_negative:
            return -log_likelihood
        return log_likelihood


class AsymmetricGaussianLikelihood(LikelihoodModel):
    """Asymmetric Gaussian likelihood model."""

    def __init__(self, return_negative: bool = True):
        """
        Initialize asymmetric Gaussian likelihood.

        Args:
            return_negative: If True, return -log(L) for minimization
        """
        self.return_negative = return_negative

    def compute(self, computed_values: Dict[str, float], observables: list) -> float:
        """
        Compute asymmetric Gaussian log-likelihood.

        Uses sigma_lower when computed < observed, sigma_upper otherwise.

        Args:
            computed_values: Dict mapping observable name to computed value
            observables: List of ObservableConfig with observed values

        Returns:
            Log-likelihood (or negative log-likelihood if return_negative=True)
        """
        chi2 = 0.0

        for obs in observables:
            computed = computed_values[obs.name]
            observed = obs.observed_value
            sigma = obs.fdGetUncertainty(computed)

            chi2 += ((computed - observed) / sigma) ** 2

        log_likelihood = -0.5 * chi2

        if self.return_negative:
            return -log_likelihood
        return log_likelihood


def create_likelihood(config: dict, observables: list) -> LikelihoodModel:
    """
    Factory function to create likelihood model from config.

    Args:
        config: Likelihood configuration dict
        observables: List of ObservableConfig objects

    Returns:
        LikelihoodModel instance
    """
    return_negative = config.get('return_negative', True)

    # Auto-detect if any observables have asymmetric uncertainties
    has_asymmetric = any(obs.bIsAsymmetric for obs in observables)

    if has_asymmetric:
        return AsymmetricGaussianLikelihood(return_negative=return_negative)
    else:
        return GaussianLikelihood(return_negative=return_negative)
