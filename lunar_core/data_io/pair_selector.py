"""Conservative metadata-based mission product pairing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from lunar_core.data_io.mission_product import MissionProduct


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

    def __post_init__(self) -> None:
        if self.pair_id is None:
            self.pair_id = f"{self.source_product_id}__{self.target_product_id}"
        if self.pair_type is None:
            self.pair_type = f"{self.source_mission or 'unknown'}:{self.source_instrument or 'unknown'}__{self.target_mission or 'unknown'}:{self.target_instrument or 'unknown'}"

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
            "intersection_area": self.intersection_area,
            "source_area": self.source_area,
            "target_area": self.target_area,
            "gsd_ratio": self.gsd_ratio,
            "acquisition_time_delta_seconds": self.acquisition_time_delta_seconds,
            "pair_type": self.pair_type,
            "selection_method": self.selection_method,
            "selection_status": self.status,
            "status": self.status,
            "reason": self.reason,
        }


def _distance_degrees(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if first.center_lat_deg is None or first.center_lon_deg is None:
        return None
    if second.center_lat_deg is None or second.center_lon_deg is None:
        return None
    return ((first.center_lat_deg - second.center_lat_deg) ** 2 + (first.center_lon_deg - second.center_lon_deg) ** 2) ** 0.5


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2.0


def _clip_polygon(subject: list[tuple[float, float]], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> list[tuple[float, float]]:
    """Clip a convex polygon against one directed edge."""
    if not subject:
        return []

    def cross(first: tuple[float, float], second: tuple[float, float], point: tuple[float, float]) -> float:
        return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])

    output: list[tuple[float, float]] = []
    previous = subject[-1]
    previous_inside = cross(edge_start, edge_end, previous) >= 0
    for current in subject:
        current_inside = cross(edge_start, edge_end, current) >= 0
        if current_inside != previous_inside:
            denominator = cross(edge_start, edge_end, current) - cross(edge_start, edge_end, previous)
            if denominator:
                ratio = -cross(edge_start, edge_end, previous) / denominator
                output.append(
                    (previous[0] + ratio * (current[0] - previous[0]), previous[1] + ratio * (current[1] - previous[1]))
                )
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def _intersection_area(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    clipped = list(first)
    for index, edge_start in enumerate(second):
        clipped = _clip_polygon(clipped, edge_start, second[(index + 1) % len(second)])
        if not clipped:
            return 0.0
    return _polygon_area(clipped)


def _footprint_metrics(first: MissionProduct, second: MissionProduct) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not first.footprint or not second.footprint:
        return None, None, None, None
    first_area = _polygon_area(first.footprint)
    second_area = _polygon_area(second.footprint)
    if first_area <= 0 or second_area <= 0:
        return None, first_area, second_area, None
    intersection = _intersection_area(first.footprint, second.footprint)
    return intersection / min(first_area, second_area), intersection, first_area, second_area


def _acquisition_delta(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if not first.acquisition_time or not second.acquisition_time:
        return None
    try:
        first_time = datetime.fromisoformat(first.acquisition_time.replace("Z", "+00:00"))
        second_time = datetime.fromisoformat(second.acquisition_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    return abs((first_time - second_time).total_seconds())


def propose_pair(source: MissionProduct, target: MissionProduct, max_center_distance_deg: float = 1.0) -> ProductPair:
    """Propose a pair only when available metadata supports a safe decision."""
    distance = _distance_degrees(source, target)
    overlap_ratio, intersection_area, source_area, target_area = _footprint_metrics(source, target)
    gsd_ratio = None
    if source.gsd_m and target.gsd_m and source.gsd_m > 0 and target.gsd_m > 0:
        gsd_ratio = max(source.gsd_m, target.gsd_m) / min(source.gsd_m, target.gsd_m)

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
    )
    if overlap_ratio is not None:
        if overlap_ratio <= 0:
            return ProductPair(**base, status="rejected", reason="Footprints do not intersect.")
        return ProductPair(**base, status="candidate", reason=f"Footprint overlap is {overlap_ratio:.4f} of the smaller footprint.")
    if distance is None:
        return ProductPair(**base, status="insufficient_metadata", reason="Both products need center coordinates.")
    if distance > max_center_distance_deg:
        return ProductPair(**base, status="rejected", reason=f"Center distance {distance:.4f}° exceeds threshold.")
    return ProductPair(**base, status="candidate", reason=f"Center proximity {distance:.4f}° is within threshold; footprint overlap is unverified.")
