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
    HYSI = "HYSI"          # Chandrayaan-1 Hyper Spectral Imager (~80m)
    LRO_NAC = "LRO_NAC"    # Lunar Reconnaissance Orbiter Narrow Angle Camera (~0.5m)
    SYNTHETIC = "SYNTHETIC"


class TransformationType(str, Enum):
    TRANSLATION = "translation"
    SIMILARITY = "similarity"
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
    source_frame: str = "FULL_SOURCE_IMAGE"
    reference_frame: str = "FULL_REFERENCE_IMAGE"


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
    median_residual_pixels: float = 0.0
    p95_residual_pixels: float = 0.0
    max_residual_pixels: float = 0.0
    processing_time_ms: float = 0.0
    ground_truth_available: bool = False
    reprojection_consensus_error: Optional[float] = None

    @property
    def inlier_ratio_percent(self) -> float:
        """Inlier ratio expressed as a percentage [0.0%, 100.0%]."""
        return self.inlier_ratio * 100.0

    @property
    def meets_isro_mandate(self) -> bool:
        """Scientific validation requires independent ground truth, not reprojection consensus alone."""
        return self.ground_truth_available and self.rmse_pixels < 0.40 and self.inlier_count >= 4



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


class ValidationStatus(str, Enum):
    """Explicit scientific status categories."""
    GROUND_TRUTH_VALIDATED = "GROUND_TRUTH_VALIDATED"
    REPROJECTION_ONLY = "REPROJECTION_ONLY"
    NO_GROUND_TRUTH = "NO_GROUND_TRUTH"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    CANDIDATE_UNVERIFIED = "CANDIDATE_UNVERIFIED"
    PROXIMITY_CANDIDATE = "PROXIMITY_CANDIDATE"


@dataclass
class Footprint:
    """Planetary polygon footprint on the lunar sphere."""
    vertices: List[Tuple[float, float]]  # (longitude, latitude) pairs
    status: str = "APPROXIMATE"  # "APPROXIMATE", "AUTHORITATIVE", "UNAVAILABLE"
    crs: str = "IAU_2015_MOON"
    area_km2: Optional[float] = None
    longitude_convention: str = "POSITIVE_EAST"
    latitude_convention: str = "PLANETOCENTRIC"
    provenance: Optional[str] = None


# Match is canonical alias for KeypointMatch
Match = KeypointMatch


@dataclass
class MatchSet:
    """Typed container for keypoint correspondences at a specific pipeline stage."""
    matches: List[KeypointMatch]
    stage: str = "RAW"  # "RAW", "ANMS", "INLIER", "REFINED"
    source_frame: str = "FULL_SOURCE_IMAGE"
    reference_frame: str = "FULL_REFERENCE_IMAGE"
    coordinate_convention: str = "pixel_centers_0_indexed_col_row"
    provenance: Optional[str] = None

    def __len__(self) -> int:
        return len(self.matches)


@dataclass
class TransformEstimate:
    """Estimated geometric transform with diagnostics and condition indicators."""
    model_type: str  # "TRANSLATION", "SIMILARITY", "AFFINE", "HOMOGRAPHY"
    matrix: np.ndarray
    inlier_count: int
    inlier_ratio: float
    reprojection_consensus_error: float
    median_residual: float = 0.0
    p95_residual: float = 0.0
    max_residual: float = 0.0
    condition_number: float = 1.0
    is_plausible: bool = True
    status: str = "SUCCESS"  # "SUCCESS", "DEGENERATE_TRANSFORM", "INSUFFICIENT_MATCHES"
    source_frame: str = "FULL_SOURCE_IMAGE"
    target_frame: str = "FULL_REFERENCE_IMAGE"
    provenance: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation record distinguishing ground-truth from reprojection consensus."""
    status: ValidationStatus
    ground_truth_available: bool
    reprojection_consensus_error: float
    ground_truth_rmse: Optional[float] = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    median_residual: float = 0.0
    p95_residual: float = 0.0
    max_residual: float = 0.0
    spatial_entropy: float = 0.0
    spatial_coverage_ratio: float = 0.0
    meets_mandate: bool = False
    reason: Optional[str] = None
    provenance: Optional[str] = None


@dataclass
class EvidenceRecord:
    """Reproducible audit trail record for an executed registration."""
    input_metadata: Dict[str, Any]
    checksums: Dict[str, str]
    git_commit_sha: str
    software_version: str
    matcher_selection: str
    preprocessing_parameters: Dict[str, Any]
    candidate_pair_decision: Dict[str, Any]
    raw_matches_count: int
    filtered_matches_count: int
    refined_matches_count: int
    transform_estimate: Dict[str, Any]
    metrics: Dict[str, Any]
    validation_status: str
    warnings: List[str] = field(default_factory=list)
