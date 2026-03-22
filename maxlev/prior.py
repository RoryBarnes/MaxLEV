"""Prior distribution models for MaxLEV MAP estimation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List


class PriorModel(ABC):
    """Abstract base class for a single parameter prior."""

    @abstractmethod
    def fdLogPrior(self, dValue: float) -> float:
        """Compute log-prior for a parameter value."""
        pass


class UniformPrior(PriorModel):
    """Uniform prior (contributes zero to log-posterior)."""

    def fdLogPrior(self, dValue: float) -> float:
        return 0.0


class GaussianPrior(PriorModel):
    """Symmetric Gaussian prior."""

    def __init__(self, dMean: float, dStd: float):
        self.dMean = dMean
        self.dStd = dStd

    def fdLogPrior(self, dValue: float) -> float:
        return -0.5 * ((dValue - self.dMean) / self.dStd) ** 2


class AsymmetricGaussianPrior(PriorModel):
    """Asymmetric Gaussian prior with different upper/lower widths."""

    def __init__(self, dMean: float, dStdUpper: float, dStdLower: float):
        self.dMean = dMean
        self.dStdUpper = dStdUpper
        self.dStdLower = dStdLower

    def fdLogPrior(self, dValue: float) -> float:
        dStd = self.dStdUpper if dValue >= self.dMean else self.dStdLower
        return -0.5 * ((dValue - self.dMean) / dStd) ** 2


class EmpiricalPrior(PriorModel):
    """Empirical prior from samples file, evaluated via Gaussian KDE."""

    def __init__(self, sFilePath: str, tBounds: tuple = None,
                 dScaleFactor: float = 1.0):
        from scipy.stats import gaussian_kde
        daSamples = np.loadtxt(sFilePath) * dScaleFactor
        if tBounds is not None:
            daSamples = daSamples[(daSamples >= tBounds[0])
                                 & (daSamples <= tBounds[1])]
        self._kde = gaussian_kde(daSamples)

    def fdLogPrior(self, dValue: float) -> float:
        dDensity = float(self._kde(dValue)[0])
        if dDensity <= 0.0:
            return -np.inf
        return np.log(dDensity)


class LogUniformPrior(PriorModel):
    """Log-uniform (Jeffreys) prior: p(x) proportional to 1/x.

    Appropriate for scale parameters that span orders of magnitude.
    The parameter bounds must be strictly positive.
    """

    def fdLogPrior(self, dValue: float) -> float:
        if dValue <= 0.0:
            return -np.inf
        return -np.log(dValue)


class PriorCollection:
    """Container for per-parameter priors. Computes total log-prior."""

    def __init__(self, listPriors: List[PriorModel]):
        self.listPriors = listPriors

    def fdLogPrior(self, daTheta: np.ndarray) -> float:
        """Compute total log-prior for all parameters."""
        dTotal = 0.0
        for i, prior in enumerate(self.listPriors):
            dTotal += prior.fdLogPrior(float(daTheta[i]))
        return dTotal

    def fdNegLogPrior(self, daTheta: np.ndarray) -> float:
        """Compute negative log-prior for minimization."""
        return -self.fdLogPrior(daTheta)

    def fbHasPriors(self) -> bool:
        """Return True if any non-uniform prior exists."""
        return any(
            not isinstance(prior, UniformPrior)
            for prior in self.listPriors
        )


def flistCreatePriors(listPriorConfigs: list) -> PriorCollection:
    """Build PriorCollection from list of prior config dicts."""
    listPriors = []
    for dictPrior in listPriorConfigs:
        sType = dictPrior.get("type", "uniform")
        if sType == "uniform":
            listPriors.append(UniformPrior())
        elif sType == "gaussian":
            listPriors.append(GaussianPrior(
                dMean=dictPrior["mean"],
                dStd=dictPrior["std"],
            ))
        elif sType == "asymmetric_gaussian":
            listPriors.append(AsymmetricGaussianPrior(
                dMean=dictPrior["mean"],
                dStdUpper=dictPrior["std_upper"],
                dStdLower=dictPrior["std_lower"],
            ))
        elif sType == "log_uniform":
            listPriors.append(LogUniformPrior())
        elif sType == "empirical":
            listPriors.append(EmpiricalPrior(
                sFilePath=dictPrior["samples_file"],
                tBounds=tuple(dictPrior["bounds"]) if "bounds" in dictPrior else None,
                dScaleFactor=dictPrior.get("scale_factor", 1.0),
            ))
        else:
            raise ValueError(
                f"Unknown prior type '{sType}'. "
                f"Valid types: uniform, gaussian, asymmetric_gaussian, "
                f"log_uniform, empirical"
            )
    return PriorCollection(listPriors)
