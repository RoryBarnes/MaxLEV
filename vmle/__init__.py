"""
vmle - VPlanet Maximum Likelihood Estimation

A general-purpose MLE tool for VPlanet stellar evolution models.
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .config import VMLEConfig
from .model import VMLEModel
from .optimizer import Optimizer

__all__ = ['VMLEConfig', 'VMLEModel', 'Optimizer']
