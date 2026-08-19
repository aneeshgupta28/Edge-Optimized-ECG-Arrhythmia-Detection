"""
Utility Functions
=================
Config loading, reproducibility seeding, and logging setup.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# Logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Config

def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML config file and return as nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# Reproducibility

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy. Call before any randomised operation."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: XGBoost / sklearn take seed as a constructor argument - pass cfg.seed there.


# Path helpers

def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).parent.parent
