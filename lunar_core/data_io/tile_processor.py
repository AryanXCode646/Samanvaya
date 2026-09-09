"""
Planetary Out-of-Core Tile Processor for Massive Chandrayaan-2 GeoTIFFs.
Handles rasters exceeding 10k x 10k pixels using memory-bounded windowed inference.

Features:
1. `PlanetaryTileProcessor`:
   - Configurable tile size (default 1024), overlap (default 128), max RAM threshold.
   - Windowed ingestion using `rasterio.windows.Window` avoiding full raster loading.
   - Downsampled coarse Fourier-Mellin overviews for overlapping footprint identification.
2. Sliding Window Execution:
   - Windowed DenseLoFTR matching across overlapping tile pairs.
   - Local-to-global coordinate projection.
   - Spatial Non-Maximal Suppression (NMS) / deduplication on overlapping tile boundaries using cKDTree.
3. Global Projective Consensus:
   - Robust USAC-MAGSAC++ homography on merged global tie-points.
   - Global sub-pixel registration metrics calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree

from lunar_core.models import KeypointMatch, RegistrationMetrics
from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
from lunar_core.alignment.rift_matcher import ClassicalRIFTMatcher
from lunar_core.alignment.fourier_mellin import FourierMellinAligner
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.evaluation.metrics import EvaluationEngine

logger = logging.getLogger("lunar_core.tile_processor")


@dataclass
class CoarseFootprintOverlap:
    """Bounding box of overlapping region between two large rasters."""
    ref_roi: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    src_roi: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    rotation_deg: float = 0.0
    scale: float = 1.0
    dx_pixels: float = 0.0
    dy_pixels: float = 0.0
    confidence: float = 0.0


@dataclass
class TileProcessingResult:
    """Aggregated output from windowed planetary registration."""
    global_inliers: List[KeypointMatch]
    global_homography: Optional[np.ndarray]
    total_tiles: int
    processed_tiles: int
    tiles_with_matches: int
    metrics: RegistrationMetrics
    coarse_overlap: Optional[CoarseFootprintOverlap] = None
    processing_time_s: float = 0.0
    peak_ram_mb: float = 0.0


class PlanetaryTileProcessor:
    """
    Out-of-Core Windowed Inference Engine for gigapixel lunar GeoTIFF rasters.
    """

    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 128,
        max_ram_mb: float = 4096.0,
        matcher: Optional[DenseLoFTRMatcher] = None,
        dedup_radius_px: float = 4.0,
        min_inliers_per_tile: int = 2,
        global_magsac_threshold: float = 2.0,
        inference_dim: Optional[int] = 512,
    ) -> None:
        """
        Args:
            tile_size: Width and height of inference tiles (must be divisible by 8 for LoFTR).
            overlap: Overlap margin between adjacent tiles in pixels.
            max_ram_mb: Soft RAM limit threshold in megabytes.
            matcher: Optional configured DenseLoFTRMatcher instance.
            dedup_radius_px: Spatial distance threshold for seam tie-point deduplication.
            min_inliers_per_tile: Minimum inliers required to accept a tile's matches.
            global_magsac_threshold: Reprojection threshold for global USAC-MAGSAC consensus.
            inference_dim: Target resolution to resize tiles to during LoFTR forward pass (e.g. 512).
                           Prevents O(N^2) quadratic transformer stall on CPU.
        """
        if tile_size % 8 != 0:
            raise ValueError(f"tile_size ({tile_size}) must be divisible by 8 for LoFTR compatibility.")
        if overlap >= tile_size:
            raise ValueError(f"overlap ({overlap}) must be strictly less than tile_size ({tile_size}).")

        self.tile_size = tile_size
        self.overlap = overlap
        self.step_size = tile_size - overlap
        self.max_ram_mb = max_ram_mb
        self.dedup_radius_px = dedup_radius_px
        self.min_inliers_per_tile = min_inliers_per_tile
        self.global_magsac_threshold = global_magsac_threshold
        self.inference_dim = inference_dim

        self.matcher = matcher or DenseLoFTRMatcher(
            pretrained="outdoor",
            confidence_threshold=0.15,
            grid_bins=8,
            cap_per_cell=4,
        )
        self.rift_matcher = ClassicalRIFTMatcher()
        self.pc_engine = PhaseCongruencyEngine(num_scales=3, num_orientations=4)

    @staticmethod
    def _estimate_tile_pair_ram_mb(ref_shape: Tuple[int, int], src_shape: Tuple[int, int]) -> float:
        """Approximate in-memory size (MB) for float32 source/reference tiles plus normalized copies."""
        ref_bytes = int(ref_shape[0]) * int(ref_shape[1]) * 4
        src_bytes = int(src_shape[0]) * int(src_shape[1]) * 4
        # Factor 4: reference/source + intermediate normalized/resized copies.
        return float((ref_bytes + src_bytes) * 4) / (1024.0 * 1024.0)

    def _enforce_tile_memory_budget(self, ref_shape: Tuple[int, int], src_shape: Tuple[int, int]) -> float:
        est_mb = self._estimate_tile_pair_ram_mb(ref_shape, src_shape)
        if est_mb > self.max_ram_mb:
            raise MemoryError(
                f"Tile pair memory estimate {est_mb:.1f} MB exceeds configured "
                f"max_ram_mb={self.max_ram_mb:.1f}. Reduce tile_size/inference_dim."
            )
        return est_mb

    # -------------------------------------------------------------------------
    # 1. Coarse Footprint Overlap Identification via Fourier-Mellin Overviews
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_overview(
        source: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        overview_dim: int = 512,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Reads or downsamples a low-resolution overview for coarse similarity estimation.
        Returns (overview_array_float32, (full_height, full_width)).
        """
        if isinstance(source, (str, Path)):
            with rasterio.open(str(source)) as src:
                full_shape = (src.height, src.width)
                arr = src.read(
                    1,
                    out_shape=(overview_dim, overview_dim),
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(np.float32)
                p1, p99 = np.nanpercentile(arr, 1.0), np.nanpercentile(arr, 99.0)
                denom = max(float(p99 - p1), 1e-5)
                norm = np.clip((arr - p1) / denom, 0.0, 1.0)
                return norm, full_shape

        elif isinstance(source, rasterio.DatasetReader):
            full_shape = (source.height, source.width)
            arr = source.read(
                1,
                out_shape=(overview_dim, overview_dim),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            p1, p99 = np.nanpercentile(arr, 1.0), np.nanpercentile(arr, 99.0)
            denom = max(float(p99 - p1), 1e-5)
            norm = np.clip((arr - p1) / denom, 0.0, 1.0)
            return norm, full_shape

        elif isinstance(source, np.ndarray):
            full_shape = (source.shape[0], source.shape[1])
            resized = cv2.resize(source.astype(np.float32), (overview_dim, overview_dim), interpolation=cv2.INTER_AREA)
            p1, p99 = np.percentile(resized, 1.0), np.percentile(resized, 99.0)
            denom = max(float(p99 - p1), 1e-5)
            norm = np.clip((resized - p1) / denom, 0.0, 1.0)
            return norm, full_shape

        else:
            raise TypeError(f"Unsupported raster source type: {type(source)}")

    def compute_coarse_overlapping_roi(
        self,
        source_raster: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        reference_raster: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        overview_dim: int = 512,
    ) -> CoarseFootprintOverlap:
        """
        Estimates coarse geometric displacement between two rasters using downsampled
        Fourier-Mellin spectra to isolate the overlapping bounding box before tiling.
        """
        overview_src, (h_src, w_src) = self._read_overview(source_raster, overview_dim)
        overview_ref, (h_ref, w_ref) = self._read_overview(reference_raster, overview_dim)

        rot_deg, scale, dx_ov, dy_ov, conf = FourierMellinAligner.estimate_coarse_similarity(
            overview_ref, overview_src
        )

        # Scale shift parameters to full resolution
        scale_x = w_ref / float(overview_dim)
        scale_y = h_ref / float(overview_dim)
        dx_full = float(dx_ov * scale_x)
        dy_full = float(dy_ov * scale_y)

        # Bounding boxes in full raster coordinate space
        # Default intersection if alignment is approximately identity
        ref_min_x = max(0, int(round(min(0.0, dx_full))))
        ref_min_y = max(0, int(round(min(0.0, dy_full))))
        ref_max_x = min(w_ref, int(round(w_ref + max(0.0, dx_full))))
        ref_max_y = min(h_ref, int(round(h_ref + max(0.0, dy_full))))

        # If confidence is solid, tighten overlapping ROI
        if conf > 0.15:
            ref_min_x = max(0, int(round(dx_full)))
            ref_min_y = max(0, int(round(dy_full)))
            ref_max_x = min(w_ref, int(round(w_src * scale + dx_full)))
            ref_max_y = min(h_ref, int(round(h_src * scale + dy_full)))

        # Fallback to full bounds if box is degraded
        if (ref_max_x - ref_min_x < self.tile_size // 2) or (ref_max_y - ref_min_y < self.tile_size // 2):
            ref_min_x, ref_min_y = 0, 0
            ref_max_x, ref_max_y = w_ref, h_ref

        src_roi = (0, 0, w_src, h_src)
        ref_roi = (ref_min_x, ref_min_y, ref_max_x, ref_max_y)

        return CoarseFootprintOverlap(
            ref_roi=ref_roi,
            src_roi=src_roi,
            rotation_deg=rot_deg,
            scale=scale,
            dx_pixels=dx_full,
            dy_pixels=dy_full,
            confidence=conf,
        )

    # -------------------------------------------------------------------------
    # 2. Window Grid Generation using rasterio.windows.Window
    # -------------------------------------------------------------------------

    def generate_windows(
        self,
        roi_bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> Generator[Window, None, None]:
        """
        Yields sliding rasterio Window objects covering the ROI with specified overlap.
        """
        min_x, min_y, max_x, max_y = roi_bbox
        total_h, total_w = image_shape

        if total_h <= 0 or total_w <= 0:
            raise ValueError(f"image_shape must be positive, got {image_shape}")
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"roi_bbox must have positive area, got {roi_bbox}")

        min_x = max(0, min(min_x, total_w))
        max_x = max(0, min(max_x, total_w))
        min_y = max(0, min(min_y, total_h))
        max_y = max(0, min(max_y, total_h))
        if min_x >= max_x or min_y >= max_y:
            raise ValueError(f"roi_bbox does not intersect image_shape: {roi_bbox} vs {image_shape}")

        y = min_y
        while y < max_y:
            x = min_x
            height = min(self.tile_size, total_h - y)
            if height <= 0:
                break

            while x < max_x:
                width = min(self.tile_size, total_w - x)
                if width <= 0:
                    break

                yield Window(col_off=x, row_off=y, width=width, height=height)

                if x + width >= max_x or x + width >= total_w:
                    break
                x += self.step_size

            if y + height >= max_y or y + height >= total_h:
                break
            y += self.step_size

    # -------------------------------------------------------------------------
    # 3. Window Data Extraction
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_window_data(
        raster_input: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        window: Window,
    ) -> np.ndarray:
        """
        Safely extracts 2D float32 pixel data for a specific window.
        """
        if isinstance(raster_input, (str, Path)):
            with rasterio.open(str(raster_input)) as src:
                data = src.read(1, window=window).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
        elif isinstance(raster_input, rasterio.DatasetReader):
            data = raster_input.read(1, window=window).astype(np.float32)
            if raster_input.nodata is not None:
                data[data == raster_input.nodata] = np.nan
        elif isinstance(raster_input, np.ndarray):
            r_start = int(window.row_off)
            r_end = int(window.row_off + window.height)
            c_start = int(window.col_off)
            c_end = int(window.col_off + window.width)
            if (
                r_start < 0
                or c_start < 0
                or r_end > raster_input.shape[0]
                or c_end > raster_input.shape[1]
                or r_start >= r_end
                or c_start >= c_end
            ):
                raise ValueError(f"Window {window} is outside raster bounds {raster_input.shape}")
            data = raster_input[r_start:r_end, c_start:c_end].astype(np.float32)
        else:
            raise TypeError(f"Unsupported raster type: {type(raster_input)}")

        # Normalize tile dynamic range
        p1 = float(np.nanpercentile(data, 1.0)) if np.any(np.isfinite(data)) else 0.0
        p99 = float(np.nanpercentile(data, 99.0)) if np.any(np.isfinite(data)) else 1.0
        denom = max(p99 - p1, 1e-5)
        normalized = np.clip((data - p1) / denom, 0.0, 1.0)
        return np.nan_to_num(normalized, nan=0.5).astype(np.float32)

    # -------------------------------------------------------------------------
    # 4. Spatial Seam Deduplication & NMS (cKDTree)
    # -------------------------------------------------------------------------

    def deduplicate_seam_tiepoints(
        self,
        matches: List[KeypointMatch],
    ) -> List[KeypointMatch]:
        """
        Applies Spatial Non-Maximal Suppression on overlapping tile boundary seams.
        Eliminates duplicate tie-points within `dedup_radius_px`, retaining the highest confidence.
        """
        if len(matches) <= 1:
            return matches

        # Sort descending by confidence so top points claim their neighborhood
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        ref_coords = np.array([m.ref_xy for m in sorted_matches], dtype=np.float64)

        tree = cKDTree(ref_coords)
        pairs = tree.query_pairs(r=self.dedup_radius_px)

        suppressed = set()
        for i, j in sorted(pairs):
            # i < j implies match i has higher or equal confidence
            if i not in suppressed:
                suppressed.add(j)

        deduped = [m for idx, m in enumerate(sorted_matches) if idx not in suppressed]
        return deduped

    # -------------------------------------------------------------------------
    # 5. End-to-End Windowed Out-of-Core Processing Pipeline
    # -------------------------------------------------------------------------

    def process(
        self,
        source_raster: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        reference_raster: Union[str, Path, np.ndarray, rasterio.DatasetReader],
        estimate_coarse_overlap: bool = True,
        max_tiles: Optional[int] = None,
    ) -> TileProcessingResult:
        """
        Executes sliding window inference over large GeoTIFF pairs.

        Args:
            source_raster: Source GeoTIFF (OHRC/TMC-2) path or array.
            reference_raster: Reference GeoTIFF (LRO NAC) path or array.
            estimate_coarse_overlap: Whether to use downsampled Fourier-Mellin overviews.
            max_tiles: Optional limit on total tiles to process (useful for rapid tests).

        Returns:
            TileProcessingResult containing merged global inliers and homography H.
        """
        t_start = time.perf_counter()

        # Step 1: Query image dimensions
        _, (h_src, w_src) = self._read_overview(source_raster, overview_dim=64)
        _, (h_ref, w_ref) = self._read_overview(reference_raster, overview_dim=64)

        # Step 2: Coarse footprint identification
        coarse_overlap: Optional[CoarseFootprintOverlap] = None
        if estimate_coarse_overlap:
            try:
                coarse_overlap = self.compute_coarse_overlapping_roi(source_raster, reference_raster)
                ref_roi = coarse_overlap.ref_roi
            except Exception as e:
                logger.warning(f"Coarse overlap estimation failed ({e}), falling back to full frame.")
                ref_roi = (0, 0, w_ref, h_ref)
        else:
            ref_roi = (0, 0, w_ref, h_ref)

        # Step 3: Generate reference windows
        ref_windows = list(self.generate_windows(ref_roi, (h_ref, w_ref)))
        if max_tiles is not None and max_tiles > 0:
            ref_windows = ref_windows[:max_tiles]

        total_tiles = len(ref_windows)
        processed_tiles = 0
        tiles_with_matches = 0

        raw_global_matches: List[KeypointMatch] = []
        peak_ram_mb = 0.0

        # Open dataset handles once for the entire sliding window loop to avoid repeated I/O overhead
        open_src = rasterio.open(str(source_raster)) if isinstance(source_raster, (str, Path)) else None
        open_ref = rasterio.open(str(reference_raster)) if isinstance(reference_raster, (str, Path)) else None
        src_handle = open_src if open_src is not None else source_raster
        ref_handle = open_ref if open_ref is not None else reference_raster

        try:
            # Step 4: Iterate over tile windows without full memory allocation
            for win_ref in ref_windows:
                processed_tiles += 1

                # Compute corresponding source window accounting for coarse offset if present
                if coarse_overlap and coarse_overlap.confidence > 0.20:
                    src_x = int(round(win_ref.col_off - coarse_overlap.dx_pixels))
                    src_y = int(round(win_ref.row_off - coarse_overlap.dy_pixels))
                else:
                    src_x = int(win_ref.col_off)
                    src_y = int(win_ref.row_off)

                # Clamp source window to raster bounds
                src_x = max(0, min(w_src - self.tile_size // 4, src_x))
                src_y = max(0, min(h_src - self.tile_size // 4, src_y))
                src_w = min(win_ref.width, w_src - src_x)
                src_h = min(win_ref.height, h_src - src_y)

                if src_w < 64 or src_h < 64:
                    continue

                win_src = Window(col_off=src_x, row_off=src_y, width=src_w, height=src_h)
                peak_ram_mb = max(
                    peak_ram_mb,
                    self._enforce_tile_memory_budget(
                        (int(win_ref.height), int(win_ref.width)),
                        (int(win_src.height), int(win_src.width)),
                    ),
                )

                # Read tiles out-of-core
                tile_ref = self._read_window_data(ref_handle, win_ref)
                tile_src = self._read_window_data(src_handle, win_src)

                # Fast multi-scale inference: resize if tile exceeds inference_dim
                scale_src_x, scale_src_y = 1.0, 1.0
                scale_ref_x, scale_ref_y = 1.0, 1.0

                if self.inference_dim and (tile_src.shape[0] > self.inference_dim or tile_src.shape[1] > self.inference_dim):
                    scale_src_x = tile_src.shape[1] / float(self.inference_dim)
                    scale_src_y = tile_src.shape[0] / float(self.inference_dim)
                    scale_ref_x = tile_ref.shape[1] / float(self.inference_dim)
                    scale_ref_y = tile_ref.shape[0] / float(self.inference_dim)
                    in_src = cv2.resize(tile_src, (self.inference_dim, self.inference_dim), interpolation=cv2.INTER_AREA)
                    in_ref = cv2.resize(tile_ref, (self.inference_dim, self.inference_dim), interpolation=cv2.INTER_AREA)
                else:
                    in_src = tile_src
                    in_ref = tile_ref

                # Process window pair through dense matcher
                try:
                    tile_inliers, _, _ = self.matcher.match(in_src, in_ref)
                except Exception as e:
                    logger.debug(f"Tile matching failed at ref ({win_ref.col_off}, {win_ref.row_off}): {e}")
                    tile_inliers = []

                if len(tile_inliers) < self.min_inliers_per_tile:
                    pc_ref = self.pc_engine.compute(in_ref)
                    pc_src = self.pc_engine.compute(in_src)
                    tile_inliers = self.rift_matcher.match(
                        pc_ref.max_moment,
                        pc_ref.orientation_max_idx,
                        pc_src.max_moment,
                        pc_src.orientation_max_idx,
                    )

                if len(tile_inliers) >= self.min_inliers_per_tile:
                    tiles_with_matches += 1

                    # Map local tile coordinates to global full-raster coordinates
                    for m in tile_inliers:
                        local_src_x = m.target_xy[0] * scale_src_x
                        local_src_y = m.target_xy[1] * scale_src_y
                        local_ref_x = m.ref_xy[0] * scale_ref_x
                        local_ref_y = m.ref_xy[1] * scale_ref_y

                        g_src_x = float(local_src_x + win_src.col_off)
                        g_src_y = float(local_src_y + win_src.row_off)
                        g_ref_x = float(local_ref_x + win_ref.col_off)
                        g_ref_y = float(local_ref_y + win_ref.row_off)

                        raw_global_matches.append(
                            KeypointMatch(
                                ref_xy=(g_ref_x, g_ref_y),
                                target_xy=(g_src_x, g_src_y),
                                confidence=m.confidence,
                                residual_error=m.residual_error,
                                subpixel_refined=m.subpixel_refined,
                            )
                        )

                # Explicit garbage collection to prevent memory spikes
                del tile_ref, tile_src
                if processed_tiles % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if open_src is not None:
                open_src.close()
            if open_ref is not None:
                open_ref.close()

        # Step 5: Spatial Seam Deduplication & NMS across adjacent overlapping tiles
        deduped_matches = self.deduplicate_seam_tiepoints(raw_global_matches)

        # Step 6: Global USAC-MAGSAC consensus over merged tie-points
        global_inliers: List[KeypointMatch] = []
        global_H: Optional[np.ndarray] = None

        if len(deduped_matches) >= 4:
            src_pts = np.array([m.target_xy for m in deduped_matches], dtype=np.float32)
            ref_pts = np.array([m.ref_xy for m in deduped_matches], dtype=np.float32)

            H, mask = cv2.findHomography(
                src_pts,
                ref_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.global_magsac_threshold,
                confidence=0.999,
                maxIters=5000,
            )

            if H is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                for idx, is_inlier in enumerate(inlier_mask):
                    if is_inlier:
                        global_inliers.append(deduped_matches[idx])
                global_H = H
            else:
                global_inliers = deduped_matches

        # Step 7: Global Registration Metrics
        metrics = EvaluationEngine.evaluate(
            total_matches=len(raw_global_matches),
            inliers=global_inliers,
            image_shape=(h_ref, w_ref),
            homography=global_H,
            processing_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

        elapsed = time.perf_counter() - t_start
        peak_ram_mb = 0.0
        try:
            import resource
            peak_ram_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:
            pass

        return TileProcessingResult(
            global_inliers=global_inliers,
            global_homography=global_H,
            total_tiles=total_tiles,
            processed_tiles=processed_tiles,
            tiles_with_matches=tiles_with_matches,
            metrics=metrics,
            coarse_overlap=coarse_overlap,
            processing_time_s=elapsed,
            peak_ram_mb=peak_ram_mb,
        )
