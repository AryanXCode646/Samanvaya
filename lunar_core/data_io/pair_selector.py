"""Conservative metadata-based mission product pairing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from lunar_core.data_io.mission_product import MissionProduct


from lunar_core.data_io.lunar_geometry import (
    spherical_distance_km,
    spherical_polygon_area_km2,
    spherical_polygon_intersection_area_km2,
)


@dataclass
class ProductPair:
    source_product_id: str
    target_product_id: str
    source_mission: Optional[str]
    target_mission: Optional[str]
    source_instrument: Optional[str]
    target_instrument: Optional[str]
    overlap_ratio: Optional[float]
    gsd_ratio: Optional[float]
    status: str
    reason: str
    source_product: Optional[MissionProduct] = None
    target_product: Optional[MissionProduct] = None
    pair_id: Optional[str] = None
    intersection_area: Optional[float] = None
    source_area: Optional[float] = None
    target_area: Optional[float] = None
    acquisition_time_delta_seconds: Optional[float] = None
    pair_type: Optional[str] = None
    selection_method: str = "metadata"
    overlap_status: str = "UNKNOWN"
    geometry_method: str = "unknown"
    overlap_confidence: Optional[float] = None
    illumination_delta_deg: Optional[float] = None
    metadata_quality: Optional[float] = None
    candidate_class: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pair_id is None:
            self.pair_id = f"{self.source_product_id}__{self.target_product_id}"
        if self.pair_type is None:
            self.pair_type = f"{self.source_mission or 'unknown'}:{self.source_instrument or 'unknown'}__{self.target_mission or 'unknown'}:{self.target_instrument or 'unknown'}"
        if self.candidate_class is None:
            self.candidate_class = self.status.upper()

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "source_product_id": self.source_product_id,
            "target_product_id": self.target_product_id,
            "source_mission": self.source_mission,
            "target_mission": self.target_mission,
            "source_instrument": self.source_instrument,
            "target_instrument": self.target_instrument,
            "overlap_ratio": self.overlap_ratio,
            "overlap_confidence": self.overlap_confidence,
            "intersection_area_km2": self.intersection_area,
            "source_area_km2": self.source_area,
            "target_area_km2": self.target_area,
            "gsd_ratio": self.gsd_ratio,
            "scale_ratio": self.gsd_ratio,
            "illumination_delta_deg": self.illumination_delta_deg,
            "metadata_quality": self.metadata_quality,
            "candidate_class": self.candidate_class,
            "acquisition_time_delta_seconds": self.acquisition_time_delta_seconds,
            "pair_type": self.pair_type,
            "selection_method": self.selection_method,
            "selection_status": self.status,
            "pair_status": self.status,
            "status": self.status,
            "overlap_status": self.overlap_status,
            "geometry_method": self.geometry_method,
            "source_footprint_status": self.source_product.footprint_status if self.source_product else "UNAVAILABLE",
            "target_footprint_status": self.target_product.footprint_status if self.target_product else "UNAVAILABLE",
            "reason": self.reason,
        }


def _distance_degrees(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if first.center_lat_deg is None or first.center_lon_deg is None:
        return None
    if second.center_lat_deg is None or second.center_lon_deg is None:
        return None
    return ((first.center_lat_deg - second.center_lat_deg) ** 2 + (first.center_lon_deg - second.center_lon_deg) ** 2) ** 0.5


def _polygon_bounds(points: list[tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
    if len(points) < 3:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return min(lons), max(lons), min(lats), max(lats)


def _footprint_metrics(first: MissionProduct, second: MissionProduct) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not first.footprint or not second.footprint:
        return None, None, None, None
    if getattr(first, "footprint_status", "UNAVAILABLE") == "AUTHORITATIVE" and getattr(second, "footprint_status", "UNAVAILABLE") == "AUTHORITATIVE":
        first_area_km2 = spherical_polygon_area_km2(first.footprint)
        second_area_km2 = spherical_polygon_area_km2(second.footprint)
        if first_area_km2 <= 0 or second_area_km2 <= 0:
            return None, first_area_km2, second_area_km2, None
        intersection_km2 = spherical_polygon_intersection_area_km2(first.footprint, second.footprint)
        overlap_ratio = intersection_km2 / min(first_area_km2, second_area_km2)
        return overlap_ratio, intersection_km2, first_area_km2, second_area_km2

    f1 = _polygon_bounds(first.footprint)
    f2 = _polygon_bounds(second.footprint)
    if not f1 or not f2:
        return None, None, None, None
    min_lon1, max_lon1, min_lat1, max_lat1 = f1
    min_lon2, max_lon2, min_lat2, max_lat2 = f2
    inter_min_lon = max(min_lon1, min_lon2)
    inter_max_lon = min(max_lon1, max_lon2)
    inter_min_lat = max(min_lat1, min_lat2)
    inter_max_lat = min(max_lat1, max_lat2)
    area1 = (max_lon1 - min_lon1) * (max_lat1 - min_lat1)
    area2 = (max_lon2 - min_lon2) * (max_lat2 - min_lat2)
    if area1 <= 0 or area2 <= 0:
        return None, 0.0, area1, area2
    if inter_max_lon <= inter_min_lon or inter_max_lat <= inter_min_lat:
        return 0.0, 0.0, area1, area2
    inter_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
    overlap_ratio = inter_area / min(area1, area2)
    return overlap_ratio, inter_area, area1, area2


def _acquisition_delta(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if not first.acquisition_time or not second.acquisition_time:
        return None
    try:
        first_time = datetime.fromisoformat(first.acquisition_time.replace("Z", "+00:00"))
        second_time = datetime.fromisoformat(second.acquisition_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs((first_time - second_time).total_seconds())


def _compute_illumination_delta_deg(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if (
        first.sun_azimuth_deg is None
        or first.sun_elevation_deg is None
        or second.sun_azimuth_deg is None
        or second.sun_elevation_deg is None
    ):
        return None
    az1, el1 = math.radians(first.sun_azimuth_deg), math.radians(first.sun_elevation_deg)
    az2, el2 = math.radians(second.sun_azimuth_deg), math.radians(second.sun_elevation_deg)
    # Unit solar vectors: x = cos(el)*sin(az), y = cos(el)*cos(az), z = sin(el)
    v1 = (math.cos(el1) * math.sin(az1), math.cos(el1) * math.cos(az1), math.sin(el1))
    v2 = (math.cos(el2) * math.sin(az2), math.cos(el2) * math.cos(az2), math.sin(el2))
    dot = max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3))))
    return math.degrees(math.acos(dot))


def _compute_metadata_quality(product: MissionProduct) -> float:
    score = 0.0
    total = 5.0
    if product.footprint is not None:
        score += 1.0
    if product.gsd_m is not None and product.gsd_m > 0:
        score += 1.0
    if product.sun_azimuth_deg is not None and product.sun_elevation_deg is not None:
        score += 1.0
    if product.acquisition_time is not None:
        score += 1.0
    if product.crs is not None or product.projection is not None or product.center_lat_deg is not None:
        score += 1.0
    return round(score / total, 2)


def propose_pair(source: MissionProduct, target: MissionProduct, max_center_distance_deg: float = 1.0) -> ProductPair:
    """Propose a pair using spherical lunar geometry and explainable multi-factor scoring."""
    distance = _distance_degrees(source, target)
    overlap_ratio, intersection_area, source_area, target_area = _footprint_metrics(source, target)
    gsd_ratio = None
    if source.gsd_m and target.gsd_m and source.gsd_m > 0 and target.gsd_m > 0:
        gsd_ratio = max(source.gsd_m, target.gsd_m) / min(source.gsd_m, target.gsd_m)

    illum_delta = _compute_illumination_delta_deg(source, target)
    meta_quality = round(0.5 * (_compute_metadata_quality(source) + _compute_metadata_quality(target)), 2)

    base = dict(
        source_product_id=source.product_id,
        target_product_id=target.product_id,
        source_mission=source.mission,
        target_mission=target.mission,
        source_instrument=source.instrument,
        target_instrument=target.instrument,
        overlap_ratio=overlap_ratio,
        gsd_ratio=gsd_ratio,
        source_product=source,
        target_product=target,
        intersection_area=intersection_area,
        source_area=source_area,
        target_area=target_area,
        acquisition_time_delta_seconds=_acquisition_delta(source, target),
        selection_method="footprint_intersection" if overlap_ratio is not None else "center_proximity_prefilter",
        illumination_delta_deg=illum_delta,
        metadata_quality=meta_quality,
    )

    if overlap_ratio is not None:
        approximate_geometry = source.footprint_status != "AUTHORITATIVE" or target.footprint_status != "AUTHORITATIVE"
        overlap_conf = round(min(1.0, max(0.0, overlap_ratio)), 3)
        base["overlap_confidence"] = overlap_conf

        if overlap_ratio <= 0:
            return ProductPair(
                **base,
                status="rejected",
                candidate_class="REJECTED",
                overlap_status="NON_OVERLAPPING",
                geometry_method="lat_lon_bounding_box_prefilter",
                reason="Footprints do not intersect under the conservative geographic prefilter; authoritative overlap is unavailable.",
            )
        if approximate_geometry:
            return ProductPair(
                **base,
                status="proximity_candidate",
                candidate_class="PROXIMITY_CANDIDATE",
                overlap_status="OVERLAP_UNKNOWN",
                geometry_method="spherical_area_plus_lat_lon_prefilter",
                reason=(
                    f"Candidate overlap of {overlap_ratio:.3f} detected using approximate geographic footprint ({intersection_area:.2f}); "
                    "authoritative flight projection is required before scientific verification."
                ),
            )
        cand_class = "HIGH_CONFIDENCE" if overlap_ratio >= 0.15 and (gsd_ratio is None or gsd_ratio <= 320.0) else "CANDIDATE_UNVERIFIED"
        return ProductPair(
            **base,
            status="confirmed_overlap",
            candidate_class=cand_class,
            overlap_status="CONFIRMED_OVERLAP",
            geometry_method="authoritative_lunar_geospatial_geometry",
            reason=f"Authoritative footprint overlap is {overlap_ratio:.4f} ({intersection_area:.2f}) of smaller image.",
        )

    if distance is None:
        return ProductPair(
            **base,
            status="insufficient_metadata",
            candidate_class="GEOMETRY_UNAVAILABLE",
            overlap_status="UNKNOWN",
            geometry_method="geometry_unavailable",
            reason="Neither product has footprint geometry or center coordinates; overlap cannot be determined.",
        )

    if distance > max_center_distance_deg:
        dist_km = spherical_distance_km(source.center_lat_deg, source.center_lon_deg, target.center_lat_deg, target.center_lon_deg)
        return ProductPair(
            **base,
            status="rejected",
            candidate_class="REJECTED",
            overlap_status="UNKNOWN",
            geometry_method="unknown",
            reason=f"Center distance {distance:.4f}° ({dist_km:.1f} km) exceeds threshold ({max_center_distance_deg}°); overlap rejected.",
        )

    dist_km = spherical_distance_km(source.center_lat_deg, source.center_lon_deg, target.center_lat_deg, target.center_lon_deg)
    return ProductPair(
        **base,
        status="proximity_candidate",
        candidate_class="PROXIMITY_CANDIDATE",
        overlap_status="UNKNOWN",
        geometry_method="unknown",
        reason=f"Center proximity {distance:.4f}° ({dist_km:.1f} km) is within threshold; footprint overlap is unverified and requires explicit validation.",
    )

