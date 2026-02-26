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
    bodies: Optional[List[str]] = None

    @property
    def bIsShared(self) -> bool:
        """True when parameter is shared across multiple bodies."""
        return self.bodies is not None and len(self.bodies) > 0

    @property
    def sFileName(self) -> str:
        """Extract file name: 'star.dMass' -> 'star'"""
        if self.bIsShared:
            return self.bodies[0]
        return self.name.split('.')[0]

    @property
    def sParamName(self) -> str:
        """Extract parameter name: 'star.dMass' -> 'dMass'"""
        if self.bIsShared:
            return self.name
        parts = self.name.split('.')
        return parts[1] if len(parts) > 1 else self.name

    def flistExpandedNames(self) -> List[str]:
        """Return list of body.param names for vplanet_inference."""
        if self.bIsShared:
            return [f"{sBody}.{self.name}" for sBody in self.bodies]
        return [self.name]

    @property
    def bIsLogSpace(self) -> bool:
        """Check if parameter is in log10 space."""
        return 'dex' in self.units.lower()

    def fsSharedTag(self) -> str:
        """Return formatted shared-body annotation, or empty string."""
        if self.bIsShared:
            return f"  [shared: {', '.join(self.bodies)}]"
        return ""


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
    def bIsAsymmetric(self) -> bool:
        """Check if observable has asymmetric uncertainties."""
        return (self.uncertainty_lower is not None
                and self.uncertainty_upper is not None)

    def fdGetUncertainty(self, dComputed: float) -> float:
        """Get appropriate uncertainty based on computed value."""
        if self.bIsAsymmetric:
            if dComputed < self.observed_value:
                return self.uncertainty_lower
            return self.uncertainty_upper
        return self.uncertainty


def _flistParseParameters(listRaw: list) -> List[ParameterConfig]:
    """Parse parameter entries from raw config dict."""
    return [ParameterConfig(
        name=p['name'],
        bounds=tuple(p['bounds']),
        units=p['units'],
        description=p.get('description'),
        prior=p.get('prior'),
        bodies=p.get('bodies'),
    ) for p in listRaw]


def _flistParseOutputs(listRaw: list) -> List[OutputConfig]:
    """Parse output entries from raw config dict."""
    return [OutputConfig(
        name=o['name'],
        units=o['units'],
        conversion_factor=o.get('conversion_factor', 1.0),
        description=o.get('description'),
    ) for o in listRaw]


def _flistParseObservables(listRaw: list) -> List[ObservableConfig]:
    """Parse observable entries from raw config dict."""
    return [ObservableConfig(
        name=obs['name'],
        type=obs['type'],
        observed_value=obs['observed_value'],
        output=obs.get('output'),
        expression=obs.get('expression'),
        uncertainty=obs.get('uncertainty'),
        uncertainty_lower=obs.get('uncertainty_lower'),
        uncertainty_upper=obs.get('uncertainty_upper'),
    ) for obs in listRaw]


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
        return cls(
            name=data.get('name', 'maxlev_run'),
            vplanet=data.get('vplanet', {}),
            parameters=_flistParseParameters(data.get('parameters', [])),
            outputs=_flistParseOutputs(data.get('outputs', [])),
            observables=_flistParseObservables(data.get('observables', [])),
            likelihood=data.get('likelihood', {'type': 'gaussian'}),
            optimizer=data.get('optimizer',
                               {'algorithm': 'differential_evolution'}),
            output_settings=data.get('output', {}),
        )

    def validate(self) -> List[Tuple[str, int]]:
        """
        Validate configuration.

        Returns:
            List of (error_message, line_number) tuples. Empty if valid.
        """
        listErrors = []
        listErrors.extend(self._flistValidateBasic())
        listErrors.extend(self._flistValidateObservables())
        listErrors.extend(self._flistValidateSharedParams())
        listErrors.extend(self._flistValidateVplanet())
        return listErrors

    def _flistValidateBasic(self) -> List[Tuple[str, int]]:
        """Check that required sections are non-empty."""
        listErrors = []
        if not self.parameters:
            listErrors.append(("No parameters defined", 0))
        if not self.outputs:
            listErrors.append(("No outputs defined", 0))
        if not self.observables:
            listErrors.append(("No observables defined", 0))
        return listErrors

    def _flistValidateObservables(self) -> List[Tuple[str, int]]:
        """Validate observable references and uncertainty specs."""
        listErrors = []
        setOutputNames = {o.name for o in self.outputs}
        for obs in self.observables:
            listErrors.extend(
                _flistValidateSingleObservable(obs, setOutputNames)
            )
        return listErrors

    def _flistValidateSharedParams(self) -> List[Tuple[str, int]]:
        """Validate shared parameter format and detect conflicts."""
        listErrors = []
        for param in self.parameters:
            if not param.bIsShared:
                continue
            listErrors.extend(_flistValidateSharedFormat(param))
        listErrors.extend(
            _flistValidateDuplicateBodies(self.parameters)
        )
        listErrors.extend(
            _flistValidateSharedConflicts(self.parameters)
        )
        return listErrors

    def _flistValidateVplanet(self) -> List[Tuple[str, int]]:
        """Run VPLanet-specific validation with expanded params."""
        listExpandedParams = []
        for param in self.parameters:
            for sName in param.flistExpandedNames():
                listExpandedParams.append({'name': sName})
        sExec = self.vplanet.get('executable', 'vplanet')
        dictConfig = {
            'vplanet': self.vplanet,
            'parameters': listExpandedParams,
            'outputs': [{'name': o.name} for o in self.outputs],
            'observables': [
                {'name': ob.name, 'type': ob.type,
                 'output': ob.output, 'expression': ob.expression}
                for ob in self.observables
            ],
        }
        return validation.validate_all(dictConfig, sExec)


def _flistValidateSingleObservable(obs, setOutputNames):
    """Validate one observable's type, reference, and uncertainty."""
    listErrors = []
    if obs.type == 'direct' and obs.output not in setOutputNames:
        listErrors.append((
            f"Observable '{obs.name}' references unknown "
            f"output '{obs.output}'", 0,
        ))
    if obs.type not in ['direct', 'derived']:
        listErrors.append((
            f"Observable '{obs.name}' has invalid "
            f"type '{obs.type}'", 0,
        ))
    if obs.uncertainty is None and not obs.bIsAsymmetric:
        listErrors.append((
            f"Observable '{obs.name}' has no uncertainty specified", 0,
        ))
    return listErrors


def _flistValidateSharedFormat(param):
    """Validate a single shared parameter's name and body count."""
    listErrors = []
    if '.' in param.name:
        listErrors.append((
            f"Shared parameter '{param.name}' must not use "
            f"'body.param' format (use just the option name)", 0,
        ))
    if len(param.bodies) < 2:
        listErrors.append((
            f"Shared parameter '{param.name}' needs at least "
            f"2 bodies (got {len(param.bodies)})", 0,
        ))
    listDuplicates = _flistFindDuplicates(param.bodies)
    if listDuplicates:
        listErrors.append((
            f"Shared parameter '{param.name}' has duplicate "
            f"bodies: {', '.join(listDuplicates)}", 0,
        ))
    return listErrors


def _flistFindDuplicates(listItems: list) -> List[str]:
    """Return items that appear more than once in a list."""
    setSeen = set()
    listDuplicates = []
    for sItem in listItems:
        if sItem in setSeen and sItem not in listDuplicates:
            listDuplicates.append(sItem)
        setSeen.add(sItem)
    return listDuplicates


def _flistValidateDuplicateBodies(listParameters):
    """Check for duplicate body names within each shared parameter."""
    # Already handled per-parameter in _flistValidateSharedFormat
    return []


def _flistValidateSharedConflicts(listParameters):
    """Detect conflicts between shared and non-shared params."""
    listErrors = []
    dictExpandedToParam = {}
    for param in listParameters:
        for sName in param.flistExpandedNames():
            if sName in dictExpandedToParam:
                sOther = dictExpandedToParam[sName]
                listErrors.append((
                    f"Parameter conflict: '{param.name}' and "
                    f"'{sOther}' both target '{sName}'", 0,
                ))
            else:
                dictExpandedToParam[sName] = param.name
    return listErrors
