import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BASE = "http://test.example.org/zoo#"


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture
def base_iri() -> str:
    return BASE


def pytest_collection_modifyitems(config, items):
    if shutil.which("java") is None:
        skip = pytest.mark.skip(reason="java not available for HermiT")
        for item in items:
            if "reasoner" in item.keywords:
                item.add_marker(skip)
