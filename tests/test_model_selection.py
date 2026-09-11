import numpy as np
import pytest

from lunar_core.models import KeypointMatch
from samanvaya.registration.transform import select_geometric_model


def test_select_geometric_model_pure_translation():
    # Matches with pure translation (dx=12.5, dy=-8.0)
    matches = []
    dx, dy = 12.5, -8.0
    for r in range(4):
        for c in range(5):
            sx = 10.0 + c * 20.0
            sy = 15.0 + r * 20.0
            rx = sx + dx
            ry = sy + dy
            matches.append(
                KeypointMatch(
                    ref_xy=(rx, ry),
                    target_xy=(sx, sy),
                    confidence=0.9,
                    source_frame="FULL_SOURCE_IMAGE",
                    reference_frame="FULL_REFERENCE_IMAGE",
                )
            )

    tf, diag, inliers = select_geometric_model(matches)
    assert tf is not None
    assert diag["status"] == "SUCCESS"
    assert diag["number_of_inliers"] == 20
    assert diag["reprojection_consensus_error"] < 1e-4
    # All candidate models should be evaluated
    assert "TRANSLATION" in diag["candidate_comparison"]
    assert "AFFINE" in diag["candidate_comparison"]
    assert "HOMOGRAPHY" in diag["candidate_comparison"]


def test_select_geometric_model_rejects_insufficient_matches():
    matches = [
        KeypointMatch((0.0, 0.0), (0.0, 0.0), 0.9),
        KeypointMatch((1.0, 1.0), (1.0, 1.0), 0.9),
    ]
    tf, diag, inliers = select_geometric_model(matches, min_inliers=4)
    assert tf is None
    assert diag["status"] == "INSUFFICIENT_MATCHES"


def test_select_geometric_model_handles_collinear_degeneracy():
    # 20 collinear points along a single 1D diagonal line
    matches = []
    for i in range(20):
        s = 10.0 + i * 5.0
        matches.append(KeypointMatch(ref_xy=(s + 5.0, s + 5.0), target_xy=(s, s), confidence=0.9))

    tf, diag, inliers = select_geometric_model(matches)
    assert tf is not None
    # HOMOGRAPHY cannot be fitted on collinear points; simpler TRANSLATION must be selected
    assert diag["selected_model"] == "TRANSLATION"
    assert "HOMOGRAPHY" not in diag["candidate_comparison"]
