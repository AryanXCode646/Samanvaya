"""Conservative metadata-based mission product pairing."""

from __future__ import annotations

from dataclasses import dataclass
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

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


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


def _footprint_overlap(first: MissionProduct, second: MissionProduct) -> Optional[float]:
    if not first.footprint or not second.footprint:
        return None
    first_area = _polygon_area(first.footprint)
    second_area = _polygon_area(second.footprint)
    if first_area <= 0 or second_area <= 0:
        return None
    return _intersection_area(first.footprint, second.footprint) / min(first_area, second_area)


def propose_pair(source: MissionProduct, target: MissionProduct, max_center_distance_deg: float = 1.0) -> ProductPair:
    """Propose a pair only when available metadata supports a safe decision."""
    distance = _distance_degrees(source, target)
    overlap_ratio = _footprint_overlap(source, target)
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
