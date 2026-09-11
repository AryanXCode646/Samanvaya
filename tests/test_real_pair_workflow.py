import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from samanvaya.data_io.import_real_pair import import_real_pair
from samanvaya.validation.import_control_points import import_control_points
from samanvaya.validation.run_real_pair import run_real_pair, validate_real_pair
from samanvaya.validation.evaluate_real_pair import evaluate_real_pair


def _write_tif(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, values.shape[0], 1, 1),
    ) as dst:
        dst.write(values, 1)


def test_import_real_pair_creates_pair_record_and_artifacts(tmp_path: Path):
    source_path = tmp_path / "CH2_OHRC_20240601_0001.tif"
    target_path = tmp_path / "LRO_NAC_20240601_0001.tif"
    source = np.zeros((96, 96), dtype=np.float32)
    target = np.zeros((96, 96), dtype=np.float32)
    cv = np.zeros((96, 96), dtype=np.float32)
    cv[30:70, 30:70] = 1.0
    source[30:70, 30:70] = 1.0
    target[32:72, 32:72] = 1.0
    _write_tif(source_path, source)
    _write_tif(target_path, target)

    pair_id = import_real_pair(source_path, target_path, manifest_path=tmp_path / "real_manifest.json")

    assert pair_id.startswith("PAIR_")
    manifest = json.loads((tmp_path / "real_manifest.json").read_text(encoding="utf-8"))
    assert any(entry.get("pair_id") == pair_id for entry in manifest["pairs"])


def test_control_point_import_validates_schema_and_saves_file(tmp_path: Path):
    payload = {
        "pair_id": "PAIR_TEST",
        "method": "independent_reference",
        "source": "synthetic fixture",
        "points": [
            {"source_x": 10.0, "source_y": 10.0, "reference_x": 12.0, "reference_y": 11.0, "uncertainty_px": 0.5},
            {"source_x": 50.0, "source_y": 12.0, "reference_x": 52.0, "reference_y": 13.0, "uncertainty_px": 0.5},
        ],
    }
    file_path = tmp_path / "control_points.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    imported_path = import_control_points("PAIR_TEST", file_path, manifest_path=tmp_path / "real_manifest.json")

    assert imported_path.exists()
    saved = json.loads(imported_path.read_text(encoding="utf-8"))
    assert len(saved["points"]) == 2


def test_real_pair_execution_reports_pending_ground_truth_without_control_points(tmp_path: Path):
    source_path = tmp_path / "CH2_OHRC_20240601_0002.tif"
    target_path = tmp_path / "LRO_NAC_20240601_0002.tif"
    source = np.random.default_rng(0).normal(0.5, 0.05, size=(96, 96)).astype(np.float32)
    target = np.random.default_rng(1).normal(0.5, 0.05, size=(96, 96)).astype(np.float32)
    _write_tif(source_path, source)
    _write_tif(target_path, target)

    pair_id = import_real_pair(source_path, target_path, manifest_path=tmp_path / "real_manifest.json")

    status = run_real_pair(pair_id, manifest_path=tmp_path / "real_manifest.json")

    assert status["ground_truth_status"] == "PENDING_INDEPENDENT_GROUND_TRUTH"
    assert (tmp_path / "artifacts" / "real_validation" / pair_id / "metrics.json").exists()


def test_validate_real_pair_reports_conservative_status_without_ground_truth(tmp_path: Path):
    source_path = tmp_path / "CH2_OHRC_20240601_0003.tif"
    target_path = tmp_path / "LRO_NAC_20240601_0003.tif"
    source = np.random.default_rng(0).normal(0.5, 0.05, size=(96, 96)).astype(np.float32)
    target = np.random.default_rng(1).normal(0.5, 0.05, size=(96, 96)).astype(np.float32)
    _write_tif(source_path, source)
    _write_tif(target_path, target)

    pair_id = import_real_pair(source_path, target_path, manifest_path=tmp_path / "real_manifest.json")
    status = validate_real_pair(pair_id, manifest_path=tmp_path / "real_manifest.json")

    assert status["status"] in {"NOT_VALIDATED", "FAIL"}
    assert status["ground_truth_status"] == "PENDING_INDEPENDENT_GROUND_TRUTH"
    assert "reason" in status


def test_real_pair_runner_blocks_missing_manifest_rasters(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "pairs": [{
            "pair_id": "MISSING_PAIR",
            "source_product": {"image_path": str(tmp_path / "missing_source.tif"), "gsd_m": 0.28},
            "reference_product": {"image_path": str(tmp_path / "missing_reference.tif"), "gsd_m": 0.5},
        }]
    }), encoding="utf-8")

    result = run_real_pair("MISSING_PAIR", manifest_path=manifest_path)

    assert result["status"] == "BLOCKED_BY_MISSING_DATA"
    assert result["metrics"]["rmse_px"] is None


def test_independent_checkpoints_apply_executed_transform(tmp_path: Path):
    source_path = tmp_path / "CH2_OHRC_CHECKPOINT.tif"
    target_path = tmp_path / "LRO_NAC_CHECKPOINT.tif"
    source = np.zeros((32, 32), dtype=np.float32)
    target = np.zeros((32, 32), dtype=np.float32)
    _write_tif(source_path, source)
    _write_tif(target_path, target)
    manifest_path = tmp_path / "real_manifest.json"
    pair_id = import_real_pair(source_path, target_path, manifest_path=manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_dir = Path(manifest["pairs"][0]["artifacts_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transform = [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
    (artifact_dir / "metrics.json").write_text(json.dumps({"estimated_transform": transform}), encoding="utf-8")
    control_dir = manifest_path.parent / "control_points"
    control_dir.mkdir()
    points = [
        {"source_x": 0.0, "source_y": 0.0, "reference_x": 2.0, "reference_y": 3.0},
        {"source_x": 10.0, "source_y": 0.0, "reference_x": 12.0, "reference_y": 3.0},
        {"source_x": 0.0, "source_y": 10.0, "reference_x": 2.0, "reference_y": 13.0},
        {"source_x": 10.0, "source_y": 10.0, "reference_x": 12.0, "reference_y": 13.0},
    ]
    (control_dir / f"{pair_id}.json").write_text(json.dumps({"points": points}), encoding="utf-8")

    result = evaluate_real_pair(pair_id, manifest_path=manifest_path)

    assert result["status"] == "READY"
    assert result["rmse_px"] == pytest.approx(0.0)
    assert result["evaluation_basis"] == "held_out_checkpoint_reprojection"
