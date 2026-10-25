import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def no_kaggle_credentials(monkeypatch, tmp_path):
    """No test can reach this computer's real Kaggle credentials: the variables are gone, and the
    files the Kaggle client looks for (~/.kaggle/access_token, kaggle.json) are looked for in an
    empty folder."""
    for name in [k for k in os.environ if k.startswith("KAGGLE_")]:
        monkeypatch.delenv(name)
    for name in ("HOME", "USERPROFILE", "KAGGLE_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
