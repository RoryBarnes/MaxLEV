"""
MaxLEV - VPlanet Maximum Likelihood Estimation

A general-purpose MLE tool for VPlanet stellar evolution models.
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .config import MaxLEVConfig
from .model import MaxLEVModel, AllSimulationsFailedError
from .optimizer import Optimizer

__all__ = [
    'MaxLEVConfig', 'MaxLEVModel', 'AllSimulationsFailedError', 'Optimizer'
]
