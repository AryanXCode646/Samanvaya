"""
Test Samanvaya CLI Inventory and Classification (SIH PS 26166 Phase 6 & 7).
Verifies that products are distinguished into:
AUTHORIZED_REAL, SYNTHETIC_FIXTURE, UNVERIFIED, INVALID.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from lunar_core.data_io.discovery import classify_product, inventory_products
from lunar_core.data_io.mission_product import MissionProduct, ProductStatus
from samanvaya.__main__ import main


def test_classify_product_categories(tmp_path: Path):
    # 1. Synthetic fixture
    synth_file = tmp_path / "fixtures" / "synth_sample.tif"
    synth_file.parent.mkdir(parents=True, exist_ok=True)
    synth_file.touch()
    synth_prod = MissionProduct(mission="Chandrayaan-2", instrument="OHRC", image_path=synth_file, status=ProductStatus.VALIDATED, gsd_m=0.25, width=100, height=100)
    assert classify_product(synth_prod) == "SYNTHETIC_FIXTURE"

    # 2. Invalid
    invalid_file = tmp_path / "corrupt.tif"
    invalid_file.touch()
    invalid_prod = MissionProduct(mission="Chandrayaan-2", instrument="OHRC", image_path=invalid_file, status=ProductStatus.INVALID, validation_message="Corrupted XML")
    assert classify_product(invalid_prod) == "INVALID"

    # 3. Unverified (missing required fields)
    unver_file = tmp_path / "unknown.tif"
    unver_file.touch()
    unver_prod = MissionProduct(mission="UnknownMission", instrument="Cam", image_path=unver_file, status=ProductStatus.PARSED)
    assert classify_product(unver_prod) == "UNVERIFIED"

    # 4. Authorized Real
    real_file = tmp_path / "real_missions" / "ohrc.tif"
    real_file.parent.mkdir(parents=True, exist_ok=True)
    real_file.touch()
    real_prod = MissionProduct(mission="Chandrayaan-2", instrument="OHRC", product_id="CH2_OHR_REAL", image_path=real_file, status=ProductStatus.VALIDATED, gsd_m=0.28, width=12000, height=80000)
    assert classify_product(real_prod) == "AUTHORIZED_REAL"


def test_samanvaya_cli_inventory(tmp_path: Path):
    out_file = tmp_path / "inventory.json"
    ret = main(["inventory", "--root", "data", "--output", str(out_file)])
    assert ret == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "by_classification" in data
    assert "products" in data
