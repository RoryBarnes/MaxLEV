"""Optimizer wrapper for MaxLEV."""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Callable, Tuple, Dict, Any


class Optimizer:
    """Wrapper for scipy optimizers with consistent interface."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimizer from configuration.

        Args:
            config: Optimizer configuration dictionary
        """
        self.algorithm = config.get('algorithm', 'differential_evolution')
        self.config = config

    def optimize(self, objective: Callable, bounds: np.ndarray,
                 param_names: list) -> Tuple[np.ndarray, float, dict]:
        """
        Run optimization.

        Args:
            objective: Objective function (returns negative log-likelihood)
            bounds: Parameter bounds array (N x 2)
            param_names: List of parameter names for reporting

        Returns:
            Tuple of (best_params, best_value, result_info)
        """
        print(f"\nStarting maximum likelihood optimization...")
        print(f"Method: {self.algorithm}")
        print(f"Parameter space: {len(param_names)} dimensions")

        print("\nParameter bounds:")
        for i, name in enumerate(param_names):
            print(f"  {name:30s}: [{bounds[i,0]:12.6f}, {bounds[i,1]:12.6f}]")

        if self.algorithm == 'differential_evolution':
            result = self._run_differential_evolution(objective, bounds)
        elif self.algorithm in ['powell', 'nelder-mead', 'bfgs', 'cobyla']:
            result = self._run_local_optimizer(objective, bounds)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Report results
        if result.success:
            print("\n[OK] Optimization converged successfully")
        else:
            print(f"\n[WARNING] Optimization did not fully converge: {result.message}")

        print("\nFinal parameters:")
        for i, name in enumerate(param_names):
            val = result.x[i]
            in_bounds = bounds[i,0] <= val <= bounds[i,1]
            status = "[OK]" if in_bounds else "[OUT OF BOUNDS]"
            print(f"  {name:30s} = {val:12.6e} {status}")

        return result.x, result.fun, {'success': result.success, 'message': result.message}

    def _run_differential_evolution(self, objective: Callable, bounds: np.ndarray):
        """Run differential evolution optimizer."""
        de_config = self.config.get('de_settings', {})

        result = differential_evolution(
            objective,
            bounds,
            strategy=de_config.get('strategy', 'best1bin'),
            maxiter=self.config.get('maxiter', 1000),
            popsize=de_config.get('popsize', 15),
            tol=self.config.get('tol', 0.01),
            mutation=tuple(de_config.get('mutation', [0.5, 1.0])),
            recombination=de_config.get('recombination', 0.7),
            seed=self.config.get('seed', 42),
            workers=de_config.get('workers', 1),
            updating=de_config.get('updating', 'deferred'),
            polish=de_config.get('polish', False),  # CRITICAL: disable to enforce bounds!
            disp=de_config.get('disp', True),
        )

        return result

    def _run_local_optimizer(self, objective: Callable, bounds: np.ndarray):
        """Run local optimizer (Powell, Nelder-Mead, etc.)."""
        # Start from center of bounds
        x0 = np.mean(bounds, axis=1)

        method = self.algorithm.lower()
        options = {
            'maxiter': self.config.get('maxiter', 1000),
            'disp': self.config.get('disp', True),
        }

        # Convert bounds to scipy format
        scipy_bounds = [(b[0], b[1]) for b in bounds]

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=scipy_bounds if method != 'nelder-mead' else None,
            options=options,
        )

        return result
