#!/usr/bin/env python3
"""
MaxLEV - VPlanet Maximum Likelihood Estimation

General-purpose MLE tool for VPlanet stellar evolution models.
"""

import os
import sys
import warnings

# Force single-threaded execution for numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")

from maxlev.cli import main

if __name__ == "__main__":
    main()
