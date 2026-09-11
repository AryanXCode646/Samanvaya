"""Tests for authoritative register_pair contract and RealRegistrationResult properties."""

import numpy as np
from samanvaya.validation.real_registration import (
    RealRegistrationResult,
    register_pair,
    register_products,
)


def test_real_registration_result_contract_properties():
    res = RealRegistrationResult(
        status="SUCCESS",
        source_id="CH2_OHRC_001",
        reference_id="LRO_NAC_001",
        source_instrument="OHRC",
        reference_instrument="NAC",
        overlap_status="OVERLAP_CONFIRMED",
        scale_ratio=1.78,
        illumination_metadata={"source": True, "reference": True},
        representation="phase_congruency",
        matcher="loftr",
        raw_match_count=120,
        spatially_selected_match_count=80,
        inlier_count=65,
        inlier_ratio=0.8125,
        transform_model="homography",
        transform_parameters=[[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]],
        subpixel_count=60,
        coverage_fraction=0.72,
        residual_statistics={"reprojection_rmse_px": 0.28},
        registered_output="/path/to/registered_source.tif",
        match_point_output="/path/to/matches.csv",
        validation_status="REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION",
        provenance={"commit": "abc1234"},
        source_mission="Chandrayaan-2",
        reference_mission="LRO",
    )

    assert res.source_product_id == "CH2_OHRC_001"
    assert res.reference_product_id == "LRO_NAC_001"
    assert res.source_mission == "Chandrayaan-2"
    assert res.reference_mission == "LRO"
    assert res.source_instrument == "OHRC"
    assert res.reference_instrument == "NAC"
    assert res.overlap_status == "OVERLAP_CONFIRMED"
    assert res.scale_ratio == 1.78
    assert res.raw_match_count == 120
    assert res.selected_match_count == 80
    assert res.inlier_count == 65
    assert res.inlier_ratio == 0.8125
    assert res.coverage_fraction == 0.72
    assert res.transform == [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
    assert res.subpixel_statistics["subpixel_count"] == 60
    assert res.subpixel_statistics["residual_statistics"]["reprojection_rmse_px"] == 0.28
    assert res.registered_output == "/path/to/registered_source.tif"
    assert res.match_output == "/path/to/matches.csv"
    assert res.validation_status == "REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION"
    assert res.provenance["commit"] == "abc1234"

    d = res.to_dict()
    assert d["source_product_id"] == "CH2_OHRC_001"
    assert d["reference_product_id"] == "LRO_NAC_001"
    assert d["selected_match_count"] == 80
    assert d["transform"] == [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]
    assert d["match_output"] == "/path/to/matches.csv"


def test_register_pair_is_callable():
    assert callable(register_pair)
    assert callable(register_products)
