"""
Tests for Real-Data Candidate Pair Discovery and Ranking Engine (SIH PS 26166).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from lunar_core.data_io.mission_product import MissionProduct
from samanvaya.data_io.pair_discovery import CandidatePairMatch, DiscoveryManifest, PairDiscoveryEngine


def test_candidate_pair_match_scoring_overlapping():
    engine = PairDiscoveryEngine(min_overlap_ratio=0.10)

    # Two products with high overlap (same region around 0, 0)
    p1 = MissionProduct(
        mission="Chandrayaan-2",
        instrument="OHRC",
        product_id="OHRC_001",
        image_path=Path("dummy1.tif"),
        gsd_m=0.25,
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        footprint=[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        sun_azimuth_deg=45.0,
        sun_elevation_deg=30.0,
        footprint_status="AUTHORITATIVE",
    )
    p2 = MissionProduct(
        mission="LRO",
        instrument="LROC NAC",
        product_id="LROC_001",
        image_path=Path("dummy2.tif"),
        gsd_m=0.50,
        center_lat_deg=0.01,
        center_lon_deg=0.01,
        footprint=[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
        sun_azimuth_deg=50.0,
        sun_elevation_deg=28.0,
        footprint_status="AUTHORITATIVE",
    )

    match = engine.evaluate_pair(p1, p2)
    assert match.is_valid_candidate is True
    assert match.candidate_score > 0.60
    assert match.overlap_status == "CONFIRMED_OVERLAP"
    assert match.gsd_ratio == 2.0


def test_candidate_pair_rejection_non_overlapping():
    engine = PairDiscoveryEngine(min_overlap_ratio=0.05)

    # South polar vs Equatorial
    p1 = MissionProduct(
        mission="Chandrayaan-2",
        instrument="OHRC",
        product_id="OHRC_SOUTH",
        image_path=Path("south.tif"),
        gsd_m=0.25,
        center_lat_deg=-85.0,
        center_lon_deg=0.0,
        footprint=[[-86.0, -1.0], [-84.0, -1.0], [-84.0, 1.0], [-86.0, 1.0]],
        sun_azimuth_deg=45.0,
        sun_elevation_deg=10.0,
        footprint_status="VERIFIED",
    )
    p2 = MissionProduct(
        mission="LRO",
        instrument="LROC NAC",
        product_id="LROC_EQUATOR",
        image_path=Path("eq.tif"),
        gsd_m=0.50,
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        footprint=[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        sun_azimuth_deg=45.0,
        sun_elevation_deg=70.0,
        footprint_status="VERIFIED",
    )

    match = engine.evaluate_pair(p1, p2)
    assert match.is_valid_candidate is False
    assert match.candidate_score == 0.0
    assert match.overlap_status == "NON_OVERLAPPING"
    assert match.rejection_reason is not None
    assert "NON_OVERLAPPING" in match.rejection_reason or "DISTANCE_EXCEEDED" in match.rejection_reason


def test_discovery_manifest_export_json(tmp_path: Path):
    manifest = DiscoveryManifest(
        root_directory=str(tmp_path),
        products_scanned=2,
        candidate_combinations_evaluated=1,
        valid_candidate_count=1,
        verified_overlap_count=1,
        ranked_pairs=[
            CandidatePairMatch(
                pair_id="P1__P2",
                source_product_id="P1",
                reference_product_id="P2",
                source_mission="CH2",
                reference_mission="LRO",
                source_instrument="OHRC",
                reference_instrument="NAC",
                source_path="/path/to/p1.tif",
                reference_path="/path/to/p2.tif",
                overlap_status="CONFIRMED_OVERLAP",
                geometry_method="shapely_polygon_intersection",
                overlap_ratio=0.85,
                overlap_area_km2=12.5,
                center_distance_km=1.2,
                gsd_ratio=2.0,
                sun_azimuth_delta_deg=5.0,
                sun_elevation_delta_deg=2.0,
                total_solar_delta_deg=5.4,
                acquisition_time_delta_days=10.0,
                footprint_confidence=1.0,
                candidate_score=0.92,
                is_valid_candidate=True,
            )
        ]
    )

    out_file = tmp_path / "manifest.json"
    manifest.export_json(out_file)

    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0.0"
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["pair_id"] == "P1__P2"
    assert data["pairs"][0]["candidate_score"] == 0.92
