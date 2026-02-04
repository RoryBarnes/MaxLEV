"""Observable computation from VPlanet outputs."""

import numpy as np
from typing import Dict, List


class ObservableComputer:
    """Computes observable values from VPlanet outputs."""

    def __init__(self, output_names: List[str], observables: List):
        """
        Initialize observable computer.

        Args:
            output_names: List of output parameter names (will be sorted alphabetically)
            observables: List of ObservableConfig objects
        """
        # CRITICAL: VPlanet outputs are returned alphabetically sorted!
        self.output_names = sorted(output_names)
        self.output_index = {name: i for i, name in enumerate(self.output_names)}
        self.observables = observables

    def compute(self, outputs: np.ndarray) -> Dict[str, float]:
        """
        Compute observable values from VPlanet outputs.

        Args:
            outputs: Array of output values (alphabetically sorted!)

        Returns:
            Dictionary mapping observable name to computed value
        """
        # Build namespace for expression evaluation
        namespace = {}
        for name, idx in self.output_index.items():
            namespace[name] = outputs[idx]

        results = {}
        for obs in self.observables:
            if obs.type == 'direct':
                idx = self.output_index[obs.output]
                results[obs.name] = outputs[idx]
            elif obs.type == 'derived':
                value = self._eval_expression(obs.expression, namespace)
                results[obs.name] = value

        return results

    def _eval_expression(self, expr: str, namespace: dict) -> float:
        """
        Safely evaluate a mathematical expression.

        Args:
            expr: Expression string
            namespace: Dict of variables available in expression

        Returns:
            Evaluated result
        """
        # Build safe evaluation environment
        safe_dict = {
            '__builtins__': {},
            'log10': np.log10,
            'log': np.log,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'abs': abs,
            'np': np,
        }
        safe_dict.update(namespace)

        try:
            result = eval(expr, safe_dict)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr}': {e}")
