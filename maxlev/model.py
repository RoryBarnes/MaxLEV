"""VPlanet model wrapper for MLE."""

import os
import signal
import subprocess
import numpy as np
import vplanet_inference as vpi
from typing import Optional
from .units import build_inparams_dict, build_outparams_dict


class MaxLEVModel:
    """Wrapper around vplanet_inference.VplanetModel for MLE."""

    def __init__(self, config, likelihood_model, observable_computer):
        """
        Initialize VPlanet model from configuration.

        Args:
            config: MaxLEVConfig object
            likelihood_model: LikelihoodModel instance
            observable_computer: ObservableComputer instance
        """
        self.config = config
        self.likelihood = likelihood_model
        self.observable_computer = observable_computer
        self.failure_penalty = config.likelihood.get('failure_penalty', 1e10)
        self.iTimeout = config.vplanet.get('timeout', 120)

        # Build input/output parameter dictionaries
        inparams = build_inparams_dict(config.parameters)
        outparams = build_outparams_dict(config.outputs)

        # Initialize VplanetModel
        self.vpm = vpi.VplanetModel(
            inparams,
            inpath=config.vplanet.get('inpath', '.'),
            outparams=outparams,
            executable=config.vplanet.get('executable', 'vplanet'),
            vplfile=config.vplanet.get('vplfile', 'vpl.in'),
            verbose=config.vplanet.get('verbose', False),
        )

        # Store bounds as numpy array
        self.bounds = np.array([p.bounds for p in config.parameters])
        self.param_names = [p.name for p in config.parameters]

        # Store conversion factors (sorted alphabetically to match output order)
        sorted_outputs = sorted(config.outputs, key=lambda x: x.name)
        self.daConversionFactors = np.array(
            [o.conversion_factor for o in sorted_outputs]
        )

    def check_bounds(self, theta: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        for i, val in enumerate(theta):
            if not (self.bounds[i, 0] <= val <= self.bounds[i, 1]):
                return False
        return True

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

    def run_simulation(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Run VPlanet simulation and apply unit conversion factors.

        Args:
            theta: Parameter values array

        Returns:
            Output array (alphabetically sorted!) or None if failed
        """
        fnOriginalCall = subprocess.call
        subprocess.call = self._fiTimedSubprocessCall
        iOldStderrFd = os.dup(2)
        iDevNullFd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(iDevNullFd, 2)
        try:
            outputs = self.vpm.run_model(theta, remove=True)
            outputs *= self.daConversionFactors
            return outputs
        except Exception:
            return None
        finally:
            os.dup2(iOldStderrFd, 2)
            os.close(iOldStderrFd)
            os.close(iDevNullFd)
            subprocess.call = fnOriginalCall

    def neg_log_likelihood(self, theta: np.ndarray) -> float:
        """
        Objective function for optimization.

        Returns negative log-likelihood for minimization.
        Returns failure_penalty for invalid runs.

        Args:
            theta: Parameter values array

        Returns:
            Negative log-likelihood
        """
        # Check bounds
        if not self.check_bounds(theta):
            return self.failure_penalty

        # Run simulation
        outputs = self.run_simulation(theta)
        if outputs is None:
            return self.failure_penalty

        # Check for invalid outputs
        if not np.all(np.isfinite(outputs)):
            return self.failure_penalty
        if np.any(outputs <= 0):
            return self.failure_penalty

        try:
            # Compute observables
            computed = self.observable_computer.compute(outputs)

            # Compute likelihood
            neg_log_like = self.likelihood.compute(computed, self.config.observables)

            return neg_log_like

        except Exception:
            return self.failure_penalty
