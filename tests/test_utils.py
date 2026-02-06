"""Shared test constants for MaxLEV validation tests."""

import shutil
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_PATH = str(FIXTURES_DIR)
VPLANET_EXECUTABLE = shutil.which('vplanet') or 'vplanet'
