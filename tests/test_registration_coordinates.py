from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from samanvaya.registration.coordinates import CoordinateAudit
from samanvaya.registration.windows import extract_registration_windows
from lunar_core.data_io.mission_catalog import inspect_product


def test_coordinate_audit_roundtrip():
    audit = CoordinateAudit(window_col_off=100, window_row_off=200, pyramid_scale=0.5, resampling_scale=2.0)
    assert audit.roundtrip_error(13.25, 7.5) < 1e-9
    assert audit.local_to_full(0, 0) == (100.0, 200.0)


def test_registration_windows_preserve_native_offsets(tmp_path: Path):
    source = tmp_path / "source.tif"
    reference = tmp_path / "reference.tif"
    for path in (source, reference):
        with rasterio.open(path, "w", driver="GTiff", height=16, width=12, count=1, dtype="float32", crs="EPSG:4326", transform=from_origin(0, 16, 1, 1)) as dst:
            dst.write(np.ones((1, 16, 12), dtype=np.float32))
    source_product = inspect_product(source, root_dir=tmp_path)
    reference_product = inspect_product(reference, root_dir=tmp_path)
    windows = extract_registration_windows(source_product, reference_product)
    assert windows.source_window.col_off == 0
    assert windows.reference_window.row_off == 0
    assert windows.source_image.shape == (16, 12)
    assert windows.overlap_status == "OVERLAP_UNKNOWN"
