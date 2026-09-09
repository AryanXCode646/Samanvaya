"""
Domain Data Models and Core Entities for Lunar Remote Sensing Registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple
import numpy as np


class SensorModality(str, Enum):
    OHRC = "OHRC"          # Chandrayaan-2 Orbiter High Resolution Camera (~0.25m)
    TMC2 = "TMC-2"        # Chandrayaan-2 Terrain Mapping Camera-2 (~5.0m)
    TMC2_NADIR = "CH2_TMC2_NADIR"
    TMC2_FORE = "CH2_TMC2_FORE"
    TMC2_AFT = "CH2_TMC2_AFT"
    IIRS = "IIRS"          # Chandrayaan-2 Imaging Infrared Spectrometer (~80m)
    LRO_NAC = "LRO_NAC"    # Lunar Reconnaissance Orbiter Narrow Angle Camera (~0.5m)
    SYNTHETIC = "SYNTHETIC"


class TransformationType(str, Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    THIN_PLATE_SPLINE = "thin_plate_spline"


@dataclass(frozen=True)
class SunAngles:
    """
    Planetary illumination geometry.
    """
    azimuth_deg: float      # Degrees [0, 360) clockwise from North
    elevation_deg: float    # Degrees [0, 90] above local lunar horizon

    @property
    def incidence_angle_rad(self) -> float:
        """Solar incidence angle i from local normal vector: i = 90 - elevation."""
        return np.radians(max(0.0, 90.0 - self.elevation_deg))

    @property
    def azimuth_rad(self) -> float:
        return np.radians(self.azimuth_deg)

    @property
    def sun_vector(self) -> np.ndarray:
        """Normalized 3D Cartesian sun illumination vector [sx, sy, sz]."""
        inc = self.incidence_angle_rad
        az = self.azimuth_rad
        sx = np.sin(inc) * np.sin(az)
        sy = np.sin(inc) * np.cos(az)
        sz = np.cos(inc)
        return np.array([sx, sy, sz], dtype=np.float32)


@dataclass
class GeoRaster:
    """
    Georeferenced planetary raster layer.
    """
    data: np.ndarray
    modality: SensorModality
    gsd_meters: float
    sun_angles: Optional[SunAngles] = None
    transform: Optional[Any] = None
    crs: str = "IAU2000:30100"  # Lunar sphere IAU 2000
    nodata_val: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape[:2]


@dataclass
class KeypointMatch:
    """
    Sub-pixel point correspondence between reference and target lunar frames.
    """
    ref_xy: Tuple[float, float]
    target_xy: Tuple[float, float]
    confidence: float
    subpixel_refined: bool = False
    residual_error: Optional[float] = None
    sigma_x: Optional[float] = None
    sigma_y: Optional[float] = None
    cov_xy: Optional[float] = None
    weight: Optional[float] = None


@dataclass
class RegistrationMetrics:
    """
    Quantitative performance indicators for hackathon and mission evaluation.
    """
    rmse_pixels: float
    total_matches: int
    inlier_count: int
    inlier_ratio: float
    spatial_uniformity_entropy: float  # [0.0, 1.0] Shannon entropy
    mean_residual_pixels: float = 0.0
    max_residual_pixels: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def inlier_ratio_percent(self) -> float:
        """Inlier ratio expressed as a percentage [0.0%, 100.0%]."""
        return self.inlier_ratio * 100.0

    @property
    def meets_isro_mandate(self) -> bool:
        """ISRO SIH PS 26166 mandate: Sub-pixel RMSE < 0.40 pixels with >= 4 inliers."""
        return self.rmse_pixels < 0.40 and self.inlier_count >= 4



@dataclass
class RegistrationResult:
    """
    Complete bundle produced by the lunar registration pipeline.
    """
    transformation_type: TransformationType
    transform_matrix: Optional[np.ndarray]
    matches: List[KeypointMatch]
    inliers: List[KeypointMatch]
    metrics: RegistrationMetrics
    warped_target: Optional[np.ndarray] = None
    matcher_path: str = "dense_loftr"
