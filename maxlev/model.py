"""VPlanet model wrapper for MLE."""

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

    def check_bounds(self, theta: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        for i, val in enumerate(theta):
            if not (self.bounds[i, 0] <= val <= self.bounds[i, 1]):
                return False
        return True

    def run_simulation(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Run VPlanet simulation.

        Args:
            theta: Parameter values array

        Returns:
            Output array (alphabetically sorted!) or None if failed
        """
        try:
            outputs = self.vpm.run_model(theta, remove=True)
            return outputs
        except Exception:
            return None

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
