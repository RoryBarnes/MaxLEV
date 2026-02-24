"""Configuration loading and validation for MaxLEV."""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from . import validation


@dataclass
class ParameterConfig:
    """Configuration for a single optimization parameter."""
    name: str
    bounds: Tuple[float, float]
    units: str
    description: Optional[str] = None
    prior: Optional[Dict[str, Any]] = None

    @property
    def file_name(self) -> str:
        """Extract file name: 'star.dMass' -> 'star'"""
        return self.name.split('.')[0]

    @property
    def param_name(self) -> str:
        """Extract parameter name: 'star.dMass' -> 'dMass'"""
        return self.name.split('.')[1] if '.' in self.name else self.name

    @property
    def is_log_space(self) -> bool:
        """Check if parameter is in log10 space."""
        return 'dex' in self.units.lower()


@dataclass
class OutputConfig:
    """Configuration for a VPlanet output parameter."""
    name: str
    units: str
    conversion_factor: float = 1.0
    description: Optional[str] = None


@dataclass
class ObservableConfig:
    """Configuration for an observational constraint."""
    name: str
    type: str
    observed_value: float
    output: Optional[str] = None
    expression: Optional[str] = None
    uncertainty: Optional[float] = None
    uncertainty_lower: Optional[float] = None
    uncertainty_upper: Optional[float] = None

    @property
    def is_asymmetric(self) -> bool:
        """Check if observable has asymmetric uncertainties."""
        return self.uncertainty_lower is not None and self.uncertainty_upper is not None

    def get_uncertainty(self, computed: float) -> float:
        """Get appropriate uncertainty based on computed value."""
        if self.is_asymmetric:
            return self.uncertainty_lower if computed < self.observed_value else self.uncertainty_upper
        return self.uncertainty


@dataclass
class MaxLEVConfig:
    """Main configuration container."""
    name: str
    vplanet: Dict[str, Any]
    parameters: List[ParameterConfig]
    outputs: List[OutputConfig]
    observables: List[ObservableConfig]
    likelihood: Dict[str, Any]
    optimizer: Dict[str, Any]
    output_settings: Dict[str, Any]

    @classmethod
    def from_json(cls, filepath: str) -> 'MaxLEVConfig':
        """Load configuration from JSON file."""
        pathConfigDir = Path(filepath).resolve().parent
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = cls._from_dict(data)
        config._fnResolveInpath(pathConfigDir)
        return config

    def _fnResolveInpath(self, pathConfigDir: Path) -> None:
        """Resolve vplanet inpath relative to config file directory."""
        sRawInpath = self.vplanet.get('inpath', '.')
        self.vplanet['inpath'] = str(
            (pathConfigDir / sRawInpath).resolve()
        )

    @classmethod
    def _from_dict(cls, data: dict) -> 'MaxLEVConfig':
        """Parse configuration dictionary."""
        parameters = [ParameterConfig(
            name=p['name'],
            bounds=tuple(p['bounds']),
            units=p['units'],
            description=p.get('description'),
            prior=p.get('prior'),
        ) for p in data.get('parameters', [])]

        outputs = [OutputConfig(
            name=o['name'],
            units=o['units'],
            conversion_factor=o.get('conversion_factor', 1.0),
            description=o.get('description')
        ) for o in data.get('outputs', [])]

        observables = [ObservableConfig(
            name=obs['name'],
            type=obs['type'],
            observed_value=obs['observed_value'],
            output=obs.get('output'),
            expression=obs.get('expression'),
            uncertainty=obs.get('uncertainty'),
            uncertainty_lower=obs.get('uncertainty_lower'),
            uncertainty_upper=obs.get('uncertainty_upper')
        ) for obs in data.get('observables', [])]

        return cls(
            name=data.get('name', 'maxlev_run'),
            vplanet=data.get('vplanet', {}),
            parameters=parameters,
            outputs=outputs,
            observables=observables,
            likelihood=data.get('likelihood', {'type': 'gaussian'}),
            optimizer=data.get('optimizer', {'algorithm': 'differential_evolution'}),
            output_settings=data.get('output', {})
        )

    def validate(self) -> List[Tuple[str, int]]:
        """
        Validate configuration.

        Returns:
            List of (error_message, line_number) tuples. Empty if valid.
        """
        errors = []

        # Basic validation
        if not self.parameters:
            errors.append(("No parameters defined", 0))
        if not self.outputs:
            errors.append(("No outputs defined", 0))
        if not self.observables:
            errors.append(("No observables defined", 0))

        # Validate observables reference valid outputs
        output_names = {o.name for o in self.outputs}
        for obs in self.observables:
            if obs.type == 'direct' and obs.output not in output_names:
                errors.append((f"Observable '{obs.name}' references unknown output '{obs.output}'", 0))
            if obs.type not in ['direct', 'derived']:
                errors.append((f"Observable '{obs.name}' has invalid type '{obs.type}'", 0))

            # Check uncertainty specification
            if obs.uncertainty is None and not obs.is_asymmetric:
                errors.append((f"Observable '{obs.name}' has no uncertainty specified", 0))

        # VPlanet validation
        vplanet_exec = self.vplanet.get('executable', 'vplanet')
        config_dict = {
            'vplanet': self.vplanet,
            'parameters': [{'name': p.name} for p in self.parameters],
            'outputs': [{'name': o.name} for o in self.outputs],
            'observables': [{'name': ob.name, 'type': ob.type, 'output': ob.output, 'expression': ob.expression}
                          for ob in self.observables]
        }

        vplanet_errors = validation.validate_all(config_dict, vplanet_exec)
        errors.extend(vplanet_errors)

        return errors
