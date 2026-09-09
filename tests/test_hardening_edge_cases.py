"""Focused regression tests for live-demo failure modes."""

from unittest.mock import MagicMock

import numpy as np

from lunar_core import pipeline as pipeline_module
from lunar_core.evaluation.pdf_reporter import compute_tiepoint_csv_sha256
from lunar_core.models import KeypointMatch
from lunar_core.pipeline import LunarCorePipeline


def _small_pair() -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros((64, 64), dtype=np.float32)
    target = np.zeros((64, 64), dtype=np.float32)
    reference[16:48, 20:44] = 1.0
    target[18:50, 22:46] = 1.0
    return reference, target


def test_rift_failure_returns_explicit_non_crashing_result(monkeypatch):
    reference, target = _small_pair()
    pipeline = LunarCorePipeline(enable_photometric=False, enable_subpixel=False)
    pipeline.dense_matcher.match_patches = lambda _reference, _target: []

    class FailingRift:
        def match(self, *_args, **_kwargs):
            raise RuntimeError("simulated RIFT failure")

    monkeypatch.setattr(pipeline_module, "ClassicalRIFTMatcher", FailingRift)
    result = pipeline.register(reference, target)

    assert result.matcher_path == "classical_rift_failed"
    assert result.inliers == []
    assert result.transform_matrix is None


def test_cuda_oom_triggers_rift_fallback(monkeypatch):
    reference, target = _small_pair()
    pipeline = LunarCorePipeline(enable_photometric=False, enable_subpixel=False)
    pipeline.dense_matcher.match_patches = lambda _reference, _target: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory")
    )

    class EmptyRift:
        def match(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(pipeline_module, "ClassicalRIFTMatcher", EmptyRift)
    result = pipeline.register(reference, target)

    assert result.matcher_path == "classical_rift"
    assert result.inliers == []


def test_tiepoint_checksum_is_stable_and_content_based():
    matches = [
        KeypointMatch(
            ref_xy=(10.0, 20.0),
            target_xy=(11.0, 21.0),
            confidence=0.9,
            residual_error=0.125,
        )
    ]

    first = compute_tiepoint_csv_sha256(matches)
    second = compute_tiepoint_csv_sha256(list(matches))
    changed = compute_tiepoint_csv_sha256(
        [KeypointMatch(ref_xy=(10.0, 20.0), target_xy=(12.0, 21.0), confidence=0.9, residual_error=0.125)]
    )

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_missing_crs_uses_lunar_fallback(monkeypatch, tmp_path):
    from lunar_core.data_io.raster_reader import PlanetaryRasterReader

    fake_tif = tmp_path / "missing-crs.tif"
    fake_tif.touch()
    source = MagicMock()
    source.width = 32
    source.height = 32
    source.count = 1
    source.dtypes = ["float32"]
    source.transform = (1, 0, 0, 0, -1, 0)
    source.crs = None
    source.nodata = None
    source.read.return_value = np.ones((1, 32, 32), dtype=np.float32)
    source.tags.return_value = {}
    context = MagicMock()
    context.__enter__.return_value = source
    context.__exit__.return_value = False

    import rasterio
    monkeypatch.setattr(rasterio, "open", lambda *_args, **_kwargs: context)
    raster = PlanetaryRasterReader.read_geotiff(fake_tif)

    assert raster.crs == "IAU2000:30100"