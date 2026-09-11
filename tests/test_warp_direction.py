import numpy as np
import pytest

from lunar_core.models import TransformationType
from lunar_core.postprocessing.magsac import RobustEstimator


def test_warp_direction_aligns_source_features_to_reference():
    """
    Given a known source image with an impulse feature at (20, 30)
    and a known transformation T mapping (20, 30) -> (45, 55),
    warp_target_to_reference must place the feature peak at (45, 55).
    """
    # Canvas size 100x100
    source = np.zeros((100, 100), dtype=np.float32)
    source[30, 20] = 1.0  # (row=30, col=20)

    # Transform: dx = +25.0, dy = +25.0
    shift_matrix = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, 25.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Warp using authoritative source->reference transform
    warped = RobustEstimator.warp_target_to_reference(
        source, shift_matrix, TransformationType.HOMOGRAPHY, output_shape=(100, 100)
    )

    # Locate peak in warped image
    peak_y, peak_x = np.unravel_index(np.argmax(warped), warped.shape)
    assert (peak_x, peak_y) == (45, 55)
    assert warped[55, 45] == 1.0

    # Also test with affine 2x3 representation
    affine_2x3 = shift_matrix[:2, :]
    warped_affine = RobustEstimator.warp_target_to_reference(
        source, affine_2x3, TransformationType.AFFINE, output_shape=(100, 100)
    )
    peak_y_aff, peak_x_aff = np.unravel_index(np.argmax(warped_affine), warped_affine.shape)
    assert (peak_x_aff, peak_y_aff) == (45, 55)
    assert warped_affine[55, 45] == 1.0
