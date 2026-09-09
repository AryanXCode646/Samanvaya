"""Regression coverage for the classical matcher fallback."""

import numpy as np

from lunar_core.pipeline import LunarCorePipeline


def test_noise_pair_uses_classical_rift_fallback():
    rng = np.random.default_rng(1234)
    reference = rng.random((128, 128), dtype=np.float32)
    target = rng.random((128, 128), dtype=np.float32)

    pipeline = LunarCorePipeline(enable_photometric=False, enable_subpixel=False)
    pipeline.dense_matcher.match_patches = lambda _reference, _target: []

    result = pipeline.register(reference, target)

    assert result is not None
    assert result.matcher_path == "classical_rift"
