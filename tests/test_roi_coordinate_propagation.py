import numpy as np

from lunar_core.alignment.scale_space import ScaleSpaceLocalizer


def test_roi_carries_target_transform_and_scale_provenance():
    reference = np.zeros((64, 64), dtype=np.float32)
    target = np.zeros((32, 32), dtype=np.float32)
    roi = ScaleSpaceLocalizer.extract_coarse_roi(reference, target, ref_gsd=1.0, tgt_gsd=2.0)
    assert roi.target_to_common.shape == (3, 3)
    assert roi.reference_common_to_full_scale == 2.0
    assert roi.target_common_to_full_scale == 1.0
    point = np.array([roi.target_roi.shape[1] / 2.0, roi.target_roi.shape[0] / 2.0, 1.0])
    recovered = np.linalg.inv(roi.target_to_common) @ point
    recovered /= recovered[2]
    assert np.all(np.isfinite(recovered))
