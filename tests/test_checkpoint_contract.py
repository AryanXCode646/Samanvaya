import csv
from pathlib import Path

import pytest

from samanvaya.validation.checkpoints import load_checkpoints


def test_checkpoint_contract_rejects_duplicate_ids_and_missing_provenance(tmp_path: Path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["point_id", "source_x", "source_y", "reference_x", "reference_y", "provenance", "quality"])
        writer.writeheader()
        writer.writerow({"point_id": "same", "source_x": 1, "source_y": 1, "reference_x": 1, "reference_y": 1, "provenance": "manual", "quality": "A"})
        writer.writerow({"point_id": "same", "source_x": 2, "source_y": 2, "reference_x": 2, "reference_y": 2, "provenance": "manual", "quality": "A"})
    with pytest.raises(ValueError, match="Duplicate"):
        load_checkpoints(path, source_shape=(10, 10), reference_shape=(10, 10))
