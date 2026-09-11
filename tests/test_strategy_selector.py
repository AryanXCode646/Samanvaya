import numpy as np
import pytest

from lunar_core.alignment.strategy_selector import (
    ClassicalSIFTMatcher,
    MatchingStrategy,
    MatchingStrategySelector,
    PhaseCorrelationMatcher,
)
from lunar_core.models import KeypointMatch, SensorModality


def test_matching_strategy_selection_rules():
    selector = MatchingStrategySelector()

    # Rule 1: Hyperspectral -> Classical RIFT
    strat, reason = selector.select_strategy(source_modality="IIRS", reference_modality="OHRC")
    assert strat == MatchingStrategy.CLASSICAL_RIFT
    assert "Hyperspectral" in reason

    # Rule 2: Large illumination delta (> 45 deg) with learned weights -> LoFTR
    strat, reason = selector.select_strategy(
        source_modality="OHRC", reference_modality="LRO_NAC", illumination_delta_deg=65.0, has_learned_weights=True
    )
    assert strat == MatchingStrategy.DENSE_LOFTR

    # Rule 3: Large illumination delta (> 45 deg) without learned weights -> Classical RIFT
    strat, reason = selector.select_strategy(
        source_modality="OHRC", reference_modality="LRO_NAC", illumination_delta_deg=65.0, has_learned_weights=False
    )
    assert strat == MatchingStrategy.CLASSICAL_RIFT

    # Rule 4: Multi-scale resolution gap (GSD ratio > 4) -> Hybrid Cascade
    strat, reason = selector.select_strategy(
        source_modality="OHRC", reference_modality="TMC2", gsd_ratio=20.0
    )
    assert strat == MatchingStrategy.HYBRID_CASCADE


def test_classical_sift_matcher():
    sift = ClassicalSIFTMatcher()
    # Synthetic image with high-contrast blobs
    img1 = np.zeros((128, 128), dtype=np.float32)
    img1[30:50, 30:50] = 1.0
    img1[80:100, 80:100] = 0.8
    img1[20:40, 80:100] = 0.6

    # Shifted image by (5, 5)
    img2 = np.zeros((128, 128), dtype=np.float32)
    img2[35:55, 35:55] = 1.0
    img2[85:105, 85:105] = 0.8
    img2[25:45, 85:105] = 0.6

    matches = sift.match(img1, img2)
    assert isinstance(matches, list)
    for m in matches:
        assert isinstance(m, KeypointMatch)
        assert m.source_frame == "FULL_SOURCE_IMAGE"


def test_phase_correlation_matcher():
    img1 = np.random.RandomState(42).uniform(0, 1, (64, 64)).astype(np.float32)
    # Exact shift of dx=4, dy=3
    dx, dy = 4, 3
    img2 = np.roll(np.roll(img1, dy, axis=0), dx, axis=1)

    shift, response = PhaseCorrelationMatcher.estimate_translation(img1, img2)
    assert abs(shift[0] - dx) <= 1.0
    assert abs(shift[1] - dy) <= 1.0
    assert response > 0.0
