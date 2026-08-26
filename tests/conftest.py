from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def monthly_series() -> np.ndarray:
    """Deterministic multiplicative monthly series (AirPassengers-like)."""
    rng = np.random.default_rng(0)
    t = np.arange(144.0)
    return (
        (120 + 2.0 * t)
        * (1 + 0.25 * np.sin(2 * np.pi * t / 12))
        * np.exp(rng.normal(0, 0.03, 144))
    )


@pytest.fixture()
def yearly_series() -> np.ndarray:
    """Deterministic non-seasonal trending series."""
    rng = np.random.default_rng(1)
    t = np.arange(40.0)
    return 50 + 3.0 * t + rng.normal(0, 4.0, 40)
