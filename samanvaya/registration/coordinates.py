"""Explicit coordinate conversions for registration windows, tiles, and scale pyramids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from rasterio.transform import Affine


@dataclass(frozen=True)
class CoordinateAudit:
    window_col_off: float
    window_row_off: float
    pyramid_scale: float = 1.0
    resampling_scale: float = 1.0
    tile_col_off: float = 0.0
    tile_row_off: float = 0.0

    def local_to_full(self, x: float, y: float) -> Tuple[float, float]:
        """Maps local window/pyramid coordinate to full image frame."""
        full_x = (x * self.resampling_scale / self.pyramid_scale) + self.tile_col_off + self.window_col_off
        full_y = (y * self.resampling_scale / self.pyramid_scale) + self.tile_row_off + self.window_row_off
        return (float(full_x), float(full_y))

    def full_to_local(self, x: float, y: float) -> Tuple[float, float]:
        """Maps full image frame coordinate to local window/pyramid coordinate."""
        local_x = (x - self.tile_col_off - self.window_col_off) * self.pyramid_scale / self.resampling_scale
        local_y = (y - self.tile_row_off - self.window_row_off) * self.pyramid_scale / self.resampling_scale
        return (float(local_x), float(local_y))

    def tile_to_full(self, x: float, y: float) -> Tuple[float, float]:
        """Maps tile coordinate to full image coordinate."""
        return (float(x + self.tile_col_off + self.window_col_off), float(y + self.tile_row_off + self.window_row_off))

    def full_to_tile(self, x: float, y: float) -> Tuple[float, float]:
        """Maps full image coordinate to tile coordinate."""
        return (float(x - self.tile_col_off - self.window_col_off), float(y - self.tile_row_off - self.window_row_off))

    def roundtrip_error(self, x: float, y: float) -> float:
        recovered = self.full_to_local(*self.local_to_full(x, y))
        return float(max(abs(recovered[0] - x), abs(recovered[1] - y)))


def pixel_to_geographic(transform: Optional[Affine], x: float, y: float) -> Optional[Tuple[float, float]]:
    if transform is None:
        return None
    return tuple(float(value) for value in (transform * (x, y)))


def geographic_to_pixel(transform: Optional[Affine], x: float, y: float) -> Optional[Tuple[float, float]]:
    if transform is None:
        return None
    return tuple(float(value) for value in (~transform * (x, y)))
