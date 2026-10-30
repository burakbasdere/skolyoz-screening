import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skolyoz_metrics.results import load_results  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def results(root):
    """A fresh deep copy of metrics/results.json that a test may mutate."""
    return copy.deepcopy(load_results(root))
