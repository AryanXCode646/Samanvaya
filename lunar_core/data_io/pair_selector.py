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


def propose_pair(source: MissionProduct, target: MissionProduct, max_center_distance_deg: float = 1.0) -> ProductPair:
    """Propose a pair only when available metadata supports a safe decision."""
    distance = _distance_degrees(source, target)
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
        overlap_ratio=None,
        gsd_ratio=gsd_ratio,
    )
    if distance is None:
        return ProductPair(**base, status="insufficient_metadata", reason="Both products need center coordinates.")
    if distance > max_center_distance_deg:
        return ProductPair(**base, status="rejected", reason=f"Center distance {distance:.4f}° exceeds threshold.")
    return ProductPair(**base, status="candidate", reason=f"Center distance {distance:.4f}° is within threshold.")
