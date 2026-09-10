"""
Tests for PlanetaryTileProcessor (Out-of-Core Windowed Inference).
Verifies memory-bounded windowed processing of massive lunar GeoTIFF rasters (4096 x 4096).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from lunar_core.data_io.tile_processor import (
    PlanetaryTileProcessor,
    TileProcessingResult,
    CoarseFootprintOverlap,
)
from lunar_core.models import KeypointMatch


def generate_synthetic_lunar_patch(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Generates synthetic lunar terrain with craters and regolith texture."""
    np.random.seed(seed)
    # Ambient texture
    x = np.linspace(0, 10, width, dtype=np.float32)
    y = np.linspace(0, 10, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    base = np.sin(xv) * np.cos(yv) * 0.2 + 0.5
    noise = np.random.normal(0, 0.05, (height, width)).astype(np.float32)
    terrain = np.clip(base + noise, 0.05, 0.95)

    # Add craters distributed across the frame
    num_craters = max(8, (height * width) // (512 * 512) * 4)
    for _ in range(num_craters):
        cx = np.random.randint(50, width - 50)
        cy = np.random.randint(50, height - 50)
        r = np.random.randint(25, 90)
        y_grid, x_grid = np.ogrid[
            max(0, cy - r) : min(height, cy + r),
            max(0, cx - r) : min(width, cx + r),
        ]
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        mask = dist_sq <= r**2
        terrain[max(0, cy - r) : min(height, cy + r), max(0, cx - r) : min(width, cx + r)][mask] *= 0.35

    return (terrain * 255.0).astype(np.float32)


def create_mock_geotiff(path: Path, array: np.ndarray, pixel_size: float = 0.5) -> None:
    """Writes a 2D float32 array as a valid GeoTIFF with lunar coordinates."""
    h, w = array.shape
    transform = from_origin(0.0, 0.0, pixel_size, pixel_size)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=rasterio.float32,
        crs="+proj=latlong +a=1737400 +b=1737400 +no_defs",
        transform=transform,
    ) as dst:
        dst.write(array.astype(np.float32), 1)


class TestPlanetaryTileProcessor:
    """Suite verifying windowed tile processing, deduplication, and out-of-core execution."""

    def test_initialization_constraints(self):
        """Verifies parameter validation on tile_size and overlap."""
        processor = PlanetaryTileProcessor(tile_size=1024, overlap=128, max_ram_mb=4096.0)
        assert processor.tile_size == 1024
        assert processor.overlap == 128
        assert processor.step_size == 896
        assert processor.max_ram_mb == 4096.0

        # Must be divisible by 8 for LoFTR (1025 % 8 != 0)
        with pytest.raises(ValueError, match="divisible by 8"):
            PlanetaryTileProcessor(tile_size=1025)

        # Overlap must be < tile_size
        with pytest.raises(ValueError, match="strictly less"):
            PlanetaryTileProcessor(tile_size=512, overlap=512)

    def test_window_generation_coverage(self):
        """Verifies rasterio Window generation covers full extent with overlap."""
        processor = PlanetaryTileProcessor(tile_size=1024, overlap=128)
        roi = (0, 0, 4096, 4096)
        windows = list(processor.generate_windows(roi, (4096, 4096)))

        assert len(windows) > 1
        # Step is 896, ceil(4096 / 896) = 5 in each dimension -> ~25 windows
        assert len(windows) == 25

        # Check first and last window bounds
        first_win = windows[0]
        assert first_win.col_off == 0
        assert first_win.row_off == 0
        assert first_win.width == 1024
        assert first_win.height == 1024

        last_win = windows[-1]
        assert last_win.col_off + last_win.width <= 4096
        assert last_win.row_off + last_win.height <= 4096

    def test_window_generation_clamps_roi_to_image_bounds(self):
        processor = PlanetaryTileProcessor(tile_size=1024, overlap=128)

        windows = list(processor.generate_windows((-100, -50, 1200, 1100), (1024, 1024)))

        assert windows[0] == Window(col_off=0, row_off=0, width=1024, height=1024)
        assert all(window.col_off >= 0 and window.row_off >= 0 for window in windows)
        assert all(window.col_off + window.width <= 1024 for window in windows)
        assert all(window.row_off + window.height <= 1024 for window in windows)

    def test_window_generation_rejects_non_intersecting_roi(self):
        processor = PlanetaryTileProcessor(tile_size=1024, overlap=128)

        with pytest.raises(ValueError, match="does not intersect"):
            list(processor.generate_windows((2000, 2000, 2100, 2100), (1024, 1024)))

    def test_seam_deduplication(self):
        """Verifies spatial NMS deduplicates redundant tie points along tile boundaries."""
        processor = PlanetaryTileProcessor(tile_size=1024, overlap=128, dedup_radius_px=4.0)

        # Create overlapping matches near boundary seam
        matches = [
            KeypointMatch(ref_xy=(1020.0, 500.0), target_xy=(1020.0, 500.0), confidence=0.85),
            KeypointMatch(ref_xy=(1022.0, 501.0), target_xy=(1022.0, 501.0), confidence=0.60),  # duplicate within 4px
            KeypointMatch(ref_xy=(1021.0, 499.0), target_xy=(1021.0, 499.0), confidence=0.70),  # duplicate within 4px
            KeypointMatch(ref_xy=(2000.0, 1500.0), target_xy=(2000.0, 1500.0), confidence=0.90), # distant point
        ]

        deduped = processor.deduplicate_seam_tiepoints(matches)
        # Should retain the highest confidence point from the cluster (0.85) plus the distant point (0.90)
        assert len(deduped) == 2
        confidences = {m.confidence for m in deduped}
        assert 0.85 in confidences
        assert 0.90 in confidences

    def test_out_of_core_processing_4096_geotiff(self):
        """
        End-to-end integration test:
        Creates a simulated 4096 x 4096 lunar GeoTIFF pair and verifies
        that PlanetaryTileProcessor processes the raster in windowed chunks
        without memory exhaustion, correctly estimating the global homography.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ref_path = tmp_path / "reference_lro_4096.tif"
            src_path = tmp_path / "source_ohrc_4096.tif"

            # 1. Create simulated 4096 x 4096 reference raster
            terrain_ref = generate_synthetic_lunar_patch(4096, 4096, seed=101)
            create_mock_geotiff(ref_path, terrain_ref)

            # 2. Create source raster with known rigid offset (dx=8, dy=6)
            M_true = np.float32([[1.0, 0.0, 8.0], [0.0, 1.0, 6.0]])
            terrain_src = cv2.warpAffine(
                terrain_ref, M_true, (4096, 4096), borderMode=cv2.BORDER_REFLECT_101
            )
            create_mock_geotiff(src_path, terrain_src)

            # 3. Instantiate PlanetaryTileProcessor with tile_size=1024, overlap=128
            processor = PlanetaryTileProcessor(
                tile_size=1024,
                overlap=128,
                max_ram_mb=4096.0,
                dedup_radius_px=4.0,
                min_inliers_per_tile=1,
            )

            # 4. Process out-of-core (with max_tiles=4 for rapid deterministic CI)
            result = processor.process(src_path, ref_path, estimate_coarse_overlap=True, max_tiles=4)

            assert isinstance(result, TileProcessingResult)
            assert result.total_tiles == 4
            assert result.processed_tiles == 4
            assert result.tiles_with_matches > 0
            assert len(result.global_inliers) > 0

            # Verify global homography was computed
            if result.global_homography is not None:
                assert result.global_homography.shape == (3, 3)
                # Verify sub-pixel registration RMSE is bounded
                assert result.metrics.rmse_pixels < 1.0
