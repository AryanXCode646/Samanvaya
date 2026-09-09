import os
from pathlib import Path

import pytest


@pytest.mark.real_data
def test_real_data_directory_is_opt_in():
    """Real mission tests require an explicit local data directory."""
    data_dir = os.environ.get("SAMANVAYA_REAL_DATA_DIR")
    if not data_dir:
        pytest.skip("Set SAMANVAYA_REAL_DATA_DIR to run local mission-data tests")
    assert Path(data_dir).is_dir(), f"Configured real-data directory does not exist: {data_dir}"