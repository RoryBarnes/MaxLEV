"""VPlanet model wrapper for MLE."""

import os
import signal
import subprocess
import numpy as np
import vplanet_inference as vpi
from typing import Optional
from .units import fdictBuildInparams, fdictBuildOutparams


class AllSimulationsFailedError(RuntimeError):
    """Raised when all VPLanet simulations fail during optimization."""
    pass


def _fiaBuildExpansionMap(listParameters):
    """
    Build index map from expanded positions to free-parameter indices.

    For shared params with N bodies, N consecutive expanded positions all
    map back to the same free-parameter index.

    Returns:
        List[int] where map[j] = free-parameter index for expanded pos j.
        None if no expansion is needed (all params are non-shared).
    """
    iaMap = []
    bHasShared = False
    for iFree, param in enumerate(listParameters):
        iExpansionCount = len(param.flistExpandedNames())
        if iExpansionCount > 1:
            bHasShared = True
        iaMap.extend([iFree] * iExpansionCount)
    if not bHasShared:
        return None
    return iaMap


class MaxLEVModel:
    """Wrapper around vplanet_inference.VplanetModel for MLE."""

    def __init__(self, config, likelihood_model, observable_computer,
                 prior_collection=None):
        """
        Initialize VPlanet model from configuration.

        Args:
            config: MaxLEVConfig object
            likelihood_model: LikelihoodModel instance
            observable_computer: ObservableComputer instance
            prior_collection: Optional PriorCollection for MAP estimation
        """
        self.config = config
        self.likelihood = likelihood_model
        self.observable_computer = observable_computer
        self.priorCollection = prior_collection
        self.dFailurePenalty = config.likelihood.get('failure_penalty', 1e10)
        self.iTimeout = config.vplanet.get('timeout', 120)
        self._fnInitVplanetModel(config)
        self._fnInitExpansionAndBounds(config)

    def _fnInitVplanetModel(self, config) -> None:
        """Build VplanetModel with input/output parameter dicts."""
        dictInparams = fdictBuildInparams(config.parameters)
        listSortedOutputs = sorted(config.outputs, key=lambda x: x.name)
        dictOutparams = fdictBuildOutparams(listSortedOutputs)
        self.vpm = vpi.VplanetModel(
            dictInparams,
            inpath=config.vplanet.get('inpath', '.'),
            outparams=dictOutparams,
            executable=config.vplanet.get('executable', 'vplanet'),
            vplfile=config.vplanet.get('vplfile', 'vpl.in'),
            verbose=config.vplanet.get('verbose', False),
        )

    def _fnInitExpansionAndBounds(self, config) -> None:
        """Set up bounds, names, expansion map, and conversion factors."""
        self.bounds = np.array([p.bounds for p in config.parameters])
        self.listParamNames = [p.name for p in config.parameters]
        self.iaExpansionMap = _fiaBuildExpansionMap(config.parameters)
        listSortedOutputs = sorted(config.outputs, key=lambda x: x.name)
        self.daConversionFactors = np.array(
            [o.conversion_factor for o in listSortedOutputs]
        )
        self.iSimulationCount = 0
        self.iSimulationFailureCount = 0
        self.iFailureCheckWindow = config.likelihood.get(
            'failure_check_window', 10
        )

    def fbCheckBounds(self, daTheta: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        for i, dValue in enumerate(daTheta):
            if not (self.bounds[i, 0] <= dValue <= self.bounds[i, 1]):
                return False
        return True

    def _fdaExpandTheta(self, daTheta: np.ndarray) -> np.ndarray:
        """Replicate free-parameter values into expanded theta."""
        if self.iaExpansionMap is None:
            return daTheta
        return np.array([daTheta[i] for i in self.iaExpansionMap])

    def _fiTimedSubprocessCall(self, *args, **kwargs):
        """Replace subprocess.call with timeout and output suppression."""
        kwargs['stdout'] = subprocess.DEVNULL
        kwargs['stderr'] = subprocess.DEVNULL
        kwargs['preexec_fn'] = os.setsid
        proc = subprocess.Popen(*args, **kwargs)
        try:
            return proc.wait(timeout=self.iTimeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise RuntimeError("VPlanet simulation timed out")

    def _fdaSuppressedRunModel(self, daTheta: np.ndarray):
        """Run vpm.run_model with suppressed stderr and timeout."""
        fnOriginalCall = subprocess.call
        subprocess.call = self._fiTimedSubprocessCall
        iOldStderrFd = os.dup(2)
        iDevNullFd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(iDevNullFd, 2)
        try:
            return self.vpm.run_model(daTheta, remove=True)
        finally:
            os.dup2(iOldStderrFd, 2)
            os.close(iOldStderrFd)
            os.close(iDevNullFd)
            subprocess.call = fnOriginalCall

    def fdaRunSimulation(self, daTheta: np.ndarray) -> Optional[np.ndarray]:
        """
        Run VPlanet simulation and apply unit conversion factors.

        Args:
            daTheta: Parameter values array

        Returns:
            Output array (alphabetically sorted!) or None if failed
        """
        daExpandedTheta = self._fdaExpandTheta(daTheta)
        try:
            daOutputs = self._fdaSuppressedRunModel(daExpandedTheta)
            daOutputs *= self.daConversionFactors
            return daOutputs
        except Exception:
            return None

    def fdNegLogLikelihood(self, daTheta: np.ndarray) -> float:
        """
        Objective function for optimization.

        Returns negative log-likelihood for minimization.
        Returns dFailurePenalty for invalid runs.
        """
        if not self.fbCheckBounds(daTheta):
            return self.dFailurePenalty

        daOutputs = self.fdaRunSimulation(daTheta)
        if not self._fbIsValidOutput(daOutputs):
            self._fnRecordSimulationFailure()
            return self.dFailurePenalty

        self.iSimulationCount += 1
        return self._fdComputeLikelihood(daOutputs)

    def _fbIsValidOutput(self, daOutputs) -> bool:
        """Check that simulation output is usable."""
        if daOutputs is None:
            return False
        return np.all(np.isfinite(daOutputs))

    def _fdComputeLikelihood(self, daOutputs: np.ndarray) -> float:
        """Compute neg-log-likelihood from valid outputs."""
        try:
            dictComputed = self.observable_computer.compute(daOutputs)
            return self.likelihood.compute(
                dictComputed, self.config.observables
            )
        except Exception:
            return self.dFailurePenalty

    def fdNegLogPosterior(self, daTheta: np.ndarray) -> float:
        """Objective for MAP: negative log-posterior = -ln(L) - ln(prior)."""
        dNegLogLike = self.fdNegLogLikelihood(daTheta)
        if dNegLogLike >= self.dFailurePenalty:
            return self.dFailurePenalty
        if self.priorCollection is None:
            return dNegLogLike
        return dNegLogLike + self.priorCollection.fdNegLogPrior(daTheta)

    def _fnRecordSimulationFailure(self) -> None:
        """Track simulation failure and abort if all fail."""
        self.iSimulationCount += 1
        self.iSimulationFailureCount += 1
        if (self.iSimulationCount >= self.iFailureCheckWindow
                and self.iSimulationFailureCount == self.iSimulationCount):
            raise AllSimulationsFailedError(
                f"All {self.iSimulationCount} VPLanet simulations failed. "
                f"Check that vplanet runs correctly with your input files."
            )
