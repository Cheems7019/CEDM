"""Shared path setup for simulation command-line entry points."""

from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The simulation scripts are intended to be runnable from any working directory.
# Keep sibling imports (notably ``utils``) and generated-artifact locations stable
# after moving the entry points into ``simulations/``.
project_root_string = str(PROJECT_ROOT)
if project_root_string not in sys.path:
    sys.path.insert(0, project_root_string)
os.chdir(PROJECT_ROOT)
