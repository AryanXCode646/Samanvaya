"""
Real-Data Candidate Pair Discovery and Ranking Engine.
SIH PS 26166: Multi-modal, Sun angle and scale invariant image correspondence.

Scans local and archive mission products (PDS4 XML, GeoTIFF), computes geometric
footprint intersections, solar angle disparities, resolution ratios, and temporal deltas,
and ranks candidate pairs for validation without fabricating non-existent overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.data_io.mission_product import MissionProduct
from lunar_core.data_io.pair_selector import ProductPair, propose_pair
from lunar_core.data_io.lunar_geometry import spherical_distance_km


@dataclass
class CandidatePairMatch:
    """Evaluated and scored candidate product pair."""
    pair_id: str
    source_product_id: str
    reference_product_id: str
    source_mission: str
    reference_mission: str
    source_instrument: str
    reference_instrument: str
    source_path: str
    reference_path: str
    overlap_status: str
    geometry_method: str
    overlap_ratio: Optional[float]
    overlap_area_km2: Optional[float]
    center_distance_km: Optional[float]
    gsd_ratio: Optional[float]
    sun_azimuth_delta_deg: Optional[float]
    sun_elevation_delta_deg: Optional[float]
    total_solar_delta_deg: Optional[float]
    acquisition_time_delta_days: Optional[float]
    footprint_confidence: float
    candidate_score: float
    rejection_reason: Optional[str] = None
    is_valid_candidate: bool = False
    source_footprint: Optional[List[List[float]]] = None
    reference_footprint: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_product_id": self.source_product_id,
            "reference_product_id": self.reference_product_id,
            "source": {
                "mission": self.source_mission,
                "instrument": self.source_instrument,
                "file": self.source_path,
                "footprint": self.source_footprint,
            },
            "reference": {
                "mission": self.reference_mission,
                "instrument": self.reference_instrument,
                "file": self.reference_path,
                "footprint": self.reference_footprint,
            },
            "geometry": {
                "overlap_status": self.overlap_status,
                "geometry_method": self.geometry_method,
                "overlap_ratio": self.overlap_ratio,
                "overlap_area_km2": self.overlap_area_km2,
                "center_distance_km": self.center_distance_km,
                "footprint_confidence": self.footprint_confidence,
            },
            "physics": {
                "gsd_ratio": self.gsd_ratio,
                "sun_azimuth_delta_deg": self.sun_azimuth_delta_deg,
                "sun_elevation_delta_deg": self.sun_elevation_delta_deg,
                "total_solar_delta_deg": self.total_solar_delta_deg,
                "acquisition_time_delta_days": self.acquisition_time_delta_days,
            },
            "candidate_score": round(self.candidate_score, 4),
            "is_valid_candidate": self.is_valid_candidate,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class DiscoveryManifest:
    """Exportable manifest of discovered and ranked mission pairs."""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    root_directory: str = ""
    products_scanned: int = 0
    candidate_combinations_evaluated: int = 0
    valid_candidate_count: int = 0
    verified_overlap_count: int = 0
    ranked_pairs: List[CandidatePairMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "metadata": {
                "timestamp_utc": self.timestamp_utc,
                "root_directory": self.root_directory,
                "products_scanned": self.products_scanned,
                "candidate_combinations_evaluated": self.candidate_combinations_evaluated,
                "valid_candidate_count": self.valid_candidate_count,
                "verified_overlap_count": self.verified_overlap_count,
            },
            "pairs": [p.to_dict() for p in self.ranked_pairs],
        }

    def export_json(self, output_path: str | Path, indent: int = 2) -> Path:
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return out_p


class PairDiscoveryEngine:
    """
    Automated discovery, multi-factor scoring, and ranking of planetary product pairs.
    """

    def __init__(
        self,
        weight_overlap: float = 0.40,
        weight_illumination: float = 0.25,
        weight_scale: float = 0.20,
        weight_geometry: float = 0.15,
        min_overlap_ratio: float = 0.05,
        max_center_distance_km: float = 200.0,
        max_gsd_ratio: float = 350.0,
    ) -> None:
        self.w_overlap = weight_overlap
        self.w_illum = weight_illumination
        self.w_scale = weight_scale
        self.w_geom = weight_geometry
        self.min_overlap = min_overlap_ratio
        self.max_dist_km = max_center_distance_km
        self.max_gsd_ratio = max_gsd_ratio

    def scan_products(self, root_dir: str | Path) -> List[MissionProduct]:
        """Discovers and inspects all candidate mission products under directory."""
        root = Path(root_dir).expanduser().resolve()
        discovered: List[MissionProduct] = []
        extensions = {".xml", ".lbl", ".img", ".tif", ".tiff", ".qub"}
        candidate_paths = [
            p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in extensions
        ]

        seen_stems: set[str] = set()
        for p in candidate_paths:
            stem = p.stem.lower()
            if stem in seen_stems and p.suffix.lower() not in {".xml", ".lbl"}:
                continue
            try:
                prod = inspect_product(p, root_dir=root)
                if prod and prod.product_id:
                    discovered.append(prod)
                    seen_stems.add(stem)
            except Exception:
                continue

        return discovered

    def evaluate_pair(self, source: MissionProduct, reference: MissionProduct) -> CandidatePairMatch:
        """Evaluates and scores a single source-reference combination."""
        proposed: ProductPair = propose_pair(source, reference)

        # Solar angle deltas
        sun_az_delta = None
        sun_el_delta = None
        if (
            source.sun_azimuth_deg is not None and reference.sun_azimuth_deg is not None
            and source.sun_elevation_deg is not None and reference.sun_elevation_deg is not None
        ):
            sun_az_delta = abs((source.sun_azimuth_deg - reference.sun_azimuth_deg + 180.0) % 360.0 - 180.0)
            sun_el_delta = abs(source.sun_elevation_deg - reference.sun_elevation_deg)

        # Center distance
        dist_km = None
        if (
            source.center_lat_deg is not None and source.center_lon_deg is not None
            and reference.center_lat_deg is not None and reference.center_lon_deg is not None
        ):
            dist_km = spherical_distance_km(
                source.center_lat_deg, source.center_lon_deg,
                reference.center_lat_deg, reference.center_lon_deg
            )

        # Time delta
        time_days = None
        if proposed.acquisition_time_delta_seconds is not None:
            time_days = abs(proposed.acquisition_time_delta_seconds) / 86400.0

        # Quality & Confidence
        conf = proposed.overlap_confidence or 0.0
        overlap_r = proposed.overlap_ratio or 0.0
        overlap_area = proposed.intersection_area or 0.0

        # Multi-factor score computation: [0.0, 1.0]
        # 1. Overlap score: prefer higher overlap
        s_overlap = min(1.0, overlap_r)

        # 2. Illumination score: prefer moderate sun difference (tested for SIH robustness)
        s_illum = 0.5
        if proposed.illumination_delta_deg is not None:
            delta = proposed.illumination_delta_deg
            if delta <= 90.0:
                s_illum = 1.0 - (delta / 90.0) * 0.5
            else:
                s_illum = max(0.1, 1.0 - (delta / 180.0))

        # 3. Scale compatibility: prefer GSD ratio <= 20x (or 320x with cascade)
        s_scale = 0.5
        if proposed.gsd_ratio is not None and proposed.gsd_ratio > 0:
            if proposed.gsd_ratio <= 4.0:
                s_scale = 1.0
            elif proposed.gsd_ratio <= 20.0:
                s_scale = 0.8
            elif proposed.gsd_ratio <= self.max_gsd_ratio:
                s_scale = 0.5
            else:
                s_scale = 0.1

        # 4. Geometry confidence
        s_geom = 0.5
        if proposed.overlap_status == "CONFIRMED_OVERLAP":
            s_geom = 1.0
        elif proposed.overlap_status == "OVERLAP_UNKNOWN":
            s_geom = 0.6
        elif proposed.overlap_status == "NON_OVERLAPPING":
            s_geom = 0.0

        candidate_score = (
            self.w_overlap * s_overlap +
            self.w_illum * s_illum +
            self.w_scale * s_scale +
            self.w_geom * s_geom
        )

        # Validation gate
        is_valid = True
        rejection_reason = None
        if proposed.overlap_status == "NON_OVERLAPPING" or (overlap_r is not None and overlap_r <= 0.0):
            is_valid = False
            rejection_reason = "NON_OVERLAPPING: Geographic footprints do not intersect."
        elif dist_km is not None and dist_km > self.max_dist_km:
            is_valid = False
            rejection_reason = f"DISTANCE_EXCEEDED: Product centers are {dist_km:.1f} km apart (limit {self.max_dist_km} km)."
        elif proposed.gsd_ratio is not None and proposed.gsd_ratio > self.max_gsd_ratio:
            is_valid = False
            rejection_reason = f"SCALE_INCOMPATIBLE: GSD ratio {proposed.gsd_ratio:.1f}x exceeds maximum {self.max_gsd_ratio}x."

        s_fp = [[float(pt[0]), float(pt[1])] for pt in source.footprint] if source.footprint else None
        r_fp = [[float(pt[0]), float(pt[1])] for pt in reference.footprint] if reference.footprint else None

        return CandidatePairMatch(
            pair_id=f"{source.product_id}__{reference.product_id}",
            source_product_id=source.product_id or str(source.image_path),
            reference_product_id=reference.product_id or str(reference.image_path),
            source_mission=str(source.mission or "Unknown"),
            reference_mission=str(reference.mission or "Unknown"),
            source_instrument=str(source.instrument or "Unknown"),
            reference_instrument=str(reference.instrument or "Unknown"),
            source_path=str(source.image_path or source.path),
            reference_path=str(reference.image_path or reference.path),
            overlap_status=proposed.overlap_status,
            geometry_method=proposed.geometry_method,
            overlap_ratio=overlap_r if overlap_r > 0 else None,
            overlap_area_km2=overlap_area if overlap_area > 0 else None,
            center_distance_km=dist_km,
            gsd_ratio=proposed.gsd_ratio,
            sun_azimuth_delta_deg=sun_az_delta,
            sun_elevation_delta_deg=sun_el_delta,
            total_solar_delta_deg=proposed.illumination_delta_deg,
            acquisition_time_delta_days=time_days,
            footprint_confidence=conf,
            candidate_score=candidate_score if is_valid else 0.0,
            rejection_reason=rejection_reason,
            is_valid_candidate=is_valid,
            source_footprint=s_fp,
            reference_footprint=r_fp,
        )

    def discover_and_rank_pairs(self, root_dir: str | Path) -> DiscoveryManifest:
        """Executes full discovery, combinatorial pairing, scoring, and ranking."""
        products = self.scan_products(root_dir)
        evaluated_pairs: List[CandidatePairMatch] = []
        combos = 0

        # Compare every unique cross-pair
        n = len(products)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = products[i]
                p2 = products[j]
                # Filter out same-file and same-product comparisons
                if p1.image_path == p2.image_path or (p1.product_id and p1.product_id == p2.product_id):
                    continue
                combos += 1
                match_pair = self.evaluate_pair(p1, p2)
                evaluated_pairs.append(match_pair)

        # Sort pairs: valid candidates first, ranked by highest candidate_score
        ranked = sorted(
            evaluated_pairs,
            key=lambda p: (p.is_valid_candidate, p.candidate_score, p.overlap_ratio or 0.0),
            reverse=True,
        )

        valid_count = sum(1 for p in ranked if p.is_valid_candidate)
        verified_count = sum(1 for p in ranked if p.overlap_status == "CONFIRMED_OVERLAP")

        return DiscoveryManifest(
            root_directory=str(Path(root_dir).resolve()),
            products_scanned=len(products),
            candidate_combinations_evaluated=combos,
            valid_candidate_count=valid_count,
            verified_overlap_count=verified_count,
            ranked_pairs=ranked,
        )
