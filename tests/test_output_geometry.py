import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from samanvaya.validation.real_registration import _write_registered


def test_registered_output_tiff_and_georeferencing_audit(tmp_path: Path):
    """
    Test writing and reopening registered TIFF.
    Verify that dimensions, dtype, nodata, CRS, and feature coordinates are preserved.
    """
    out_tif = tmp_path / "registered_source.tif"
    h, w = 64, 80

    # Source feature at (col=20, row=30)
    data = np.zeros((h, w), dtype=np.float32)
    data[30, 20] = 100.0

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(1000.0, 2000.0, 0.5, 0.5),
        "nodata": -9999.0,
    }

    _write_registered(out_tif, data, profile)

    # Reopen and inspect
    assert out_tif.exists()
    with rasterio.open(out_tif) as ds:
        assert ds.width == w
        assert ds.height == h
        assert ds.dtypes[0] == "float32"
        assert ds.nodata == -9999.0
        assert str(ds.crs) == "EPSG:4326"
        read_data = ds.read(1)
        peak_r, peak_c = np.unravel_index(np.argmax(read_data), read_data.shape)
        assert (peak_c, peak_r) == (20, 30)
        assert read_data[30, 20] == 100.0

    # Write output_validation.json
    val_json_path = tmp_path / "output_validation.json"
    val_payload = {
        "width": w,
        "height": h,
        "dtype": "float32",
        "nodata": -9999.0,
        "crs": "EPSG:4326",
        "input_source_transform": [0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0],
        "input_reference_transform": [0.5, 0.0, 1000.0, 0.0, -0.5, 2000.0],
        "registration_transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
    val_json_path.write_text(json.dumps(val_payload, indent=2))

    # Verify JSON structure
    loaded = json.loads(val_json_path.read_text())
    assert loaded["width"] == w
    assert loaded["height"] == h
    assert len(loaded["registration_transform"]) == 3
