"""SYNTHETIC END-TO-END REGRESSION for the first real-pair control flow."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_core.data_io.mission_catalog import inspect_product
from samanvaya.validation.benchmark_real import run_real_benchmark
from samanvaya.validation.checkpoints import load_checkpoints


def _write(path: Path, values: np.ndarray) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1, dtype="float32", crs="EPSG:4326", transform=from_origin(0, values.shape[0], 1, 1)) as dst:
        dst.write(values, 1)


def test_synthetic_end_to_end_manifest_contract(tmp_path: Path):
    source = tmp_path / "CH2_OHRC_CONTRACT.tif"
    reference = tmp_path / "LRO_NAC_CONTRACT.tif"
    _write(source, np.zeros((64, 64), dtype=np.float32))
    _write(reference, np.zeros((64, 64), dtype=np.float32))
    label = """<Product_Observational><mission_name>{mission}</mission_name><instrument_id>{instrument}</instrument_id><product_id>{product}</product_id><Array_2D_Image><Axis_Array><elements>64</elements></Axis_Array><Axis_Array><elements>64</elements></Axis_Array></Array_2D_Image></Product_Observational>"""
    source.with_suffix(".xml").write_text(label.format(mission="Chandrayaan-2", instrument="OHRC", product="CH2_OHRC_CONTRACT"), encoding="utf-8")
    reference.with_suffix(".xml").write_text(label.format(mission="LRO", instrument="NAC", product="LRO_NAC_CONTRACT"), encoding="utf-8")
    source_product = inspect_product(source, root_dir=tmp_path)
    reference_product = inspect_product(reference, root_dir=tmp_path)
    assert source_product.image_path == source
    assert reference_product.image_path == reference
    checkpoints = tmp_path / "checkpoints.csv"
    with checkpoints.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["point_id", "source_x", "source_y", "reference_x", "reference_y", "provenance", "quality"])
        writer.writeheader()
        for index, coordinate in enumerate(((5, 5), (50, 5), (5, 50), (50, 50))):
            writer.writerow({"point_id": f"cp_{index}", "source_x": coordinate[0], "source_y": coordinate[1], "reference_x": coordinate[0], "reference_y": coordinate[1], "provenance": "synthetic_contract_fixture", "quality": "A"})
    assert len(load_checkpoints(checkpoints, source_shape=(64, 64), reference_shape=(64, 64))) == 4
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"pair_id": "CONTRACT", "source_product": {"image_path": str(source)}, "reference_product": {"image_path": str(reference)}, "ground_truth": {"path": str(checkpoints)}}]}), encoding="utf-8")
    result = run_real_benchmark(manifest, tmp_path / "output")
    assert result["status"] == "COMPLETE"
    assert result["results"][0]["status"] in {"SUCCESS", "NO_CORRESPONDENCE"}
