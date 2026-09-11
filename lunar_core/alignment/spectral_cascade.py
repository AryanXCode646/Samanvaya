"""
Spectral Cascade Registration Engine (SIH PS 26166).

Addresses extreme multi-modal scale disparity across:
  IIRS (80m, hyperspectral) -> TMC-2 (5m, panchromatic stereo) -> OHRC (0.25m, high-res panchromatic)

Direct matching between IIRS (80m) and OHRC (0.25m) entails a 320x spatial scale delta,
which is physically and spectrally ill-conditioned. The SpectralCascadeEngine bridges
this scale gap hierarchically:
  Stage 1: IIRS -> TMC-2 (via structural/continuum continuum band)
  Stage 2: TMC-2 -> OHRC (via high-resolution multi-scale feature matching)
  Stage 3: OHRC -> LRO NAC (or georeferenced reference base)

CRITICAL SCIENTIFIC HONESTY MANDATE:
  In-flight IIRS registration status is explicitly designated as:
  IIRS_REGISTRATION_STATUS = "EXPERIMENTAL_PARTIAL"
  Transformation composition and rigorous uncertainty propagation are mathematically
  implemented, but full end-to-end flight performance remains bound by hyperspectral
  footprint fidelity and spatial calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

IIRS_REGISTRATION_STATUS = "EXPERIMENTAL_PARTIAL"


@dataclass(frozen=True)
class CascadeStage:
    """
    Represents a single step in a multi-instrument registration cascade.
    """
    source_instrument: str
    reference_instrument: str
    transform_matrix: np.ndarray  # 3x3 projective or affine matrix
    residual_rmse: float          # In pixels of reference instrument
    scale_factor: float = 1.0     # Nominal scale ratio (e.g. gsd_src / gsd_ref)
    uncertainty_cov: Optional[np.ndarray] = None  # 2x2 spatial translation covariance
    valid: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.transform_matrix.shape != (3, 3):
            raise ValueError(f"Transform matrix must be 3x3, got {self.transform_matrix.shape}")
        if self.uncertainty_cov is not None and self.uncertainty_cov.shape != (2, 2):
            raise ValueError(f"Spatial uncertainty covariance must be 2x2, got {self.uncertainty_cov.shape}")


@dataclass
class CascadeResult:
    """
    Result of composing multi-stage registration transforms with uncertainty.
    """
    stages: List[CascadeStage]
    composed_transform: np.ndarray  # 3x3 matrix mapping source of stage 0 to ref of stage K
    effective_uncertainty_px: float # Propagated 1-sigma uncertainty in target reference frame
    composed_covariance: Optional[np.ndarray] # 2x2 composed spatial covariance
    status: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "num_stages": len(self.stages),
            "stages": [
                {
                    "source": s.source_instrument,
                    "reference": s.reference_instrument,
                    "rmse": float(s.residual_rmse),
                    "scale_factor": float(s.scale_factor),
                    "valid": bool(s.valid),
                }
                for s in self.stages
            ],
            "composed_transform": self.composed_transform.tolist(),
            "effective_uncertainty_px": float(self.effective_uncertainty_px),
            "composed_covariance": self.composed_covariance.tolist() if self.composed_covariance is not None else None,
            "notes": self.notes,
        }


class SpectralCascadeEngine:
    """
    Hierarchical transform composition and error propagation engine.
    """

    def __init__(self, designation: str = IIRS_REGISTRATION_STATUS):
        self.designation = designation

    @staticmethod
    def extract_affine_scale(H: np.ndarray) -> float:
        """
        Extract the effective linear scale factor from the 2x2 affine part of H.
        Uses the geometric mean of singular values (or square root of determinant).
        """
        A = H[:2, :2]
        det = np.linalg.det(A)
        if det <= 0:
            # Fallback to Frobenius norm / sqrt(2) if reflection or degenerate
            return float(np.linalg.norm(A, "fro") / np.sqrt(2.0))
        return float(np.sqrt(det))

    def compose(self, stages: List[CascadeStage]) -> CascadeResult:
        """
        Compose a sequence of registration stages from Stage 0 to Stage K-1.

        Convention:
          x_ref = H * x_src
          Stage 0: x_1 = H_0 * x_0
          Stage 1: x_2 = H_1 * x_1
          ...
          Stage K-1: x_K = H_{K-1} * x_{K-1}
          Composed: x_K = H_{K-1} * ... * H_1 * H_0 * x_0
        """
        if not stages:
            raise ValueError("Cannot compose an empty list of stages")

        notes: List[str] = []
        notes.append(f"Engine status: {self.designation}")

        # Check connectivity of stages
        for i in range(len(stages) - 1):
            curr_ref = stages[i].reference_instrument.upper()
            next_src = stages[i + 1].source_instrument.upper()
            if curr_ref != next_src:
                notes.append(
                    f"Warning: Stage {i} reference '{curr_ref}' does not match "
                    f"Stage {i+1} source '{next_src}'"
                )

        # 1. Compose Homographies
        composed_H = np.eye(3, dtype=np.float64)
        for stage in stages:
            if not stage.valid:
                return CascadeResult(
                    stages=stages,
                    composed_transform=np.eye(3),
                    effective_uncertainty_px=float("inf"),
                    composed_covariance=None,
                    status="FAILED_STAGE_INVALID",
                    notes=notes + [f"Stage {stage.source_instrument}->{stage.reference_instrument} is marked invalid."],
                )
            # Pre-multiply: H_next @ H_curr
            composed_H = stage.transform_matrix @ composed_H

        # Normalize composed homography so H[2, 2] == 1.0 (if non-zero)
        if abs(composed_H[2, 2]) > 1e-12:
            composed_H = composed_H / composed_H[2, 2]

        # 2. Propagate Uncertainty
        # Let stage i have spatial covariance Sigma_i (or isotropic sigma_i^2 * I)
        # In passing through subsequent stages i+1 ... K-1:
        # If x_final = A * x_i, then Cov(x_final) = A * Cov(x_i) * A^T.
        
        K = len(stages)
        composed_cov = np.zeros((2, 2), dtype=np.float64)
        variance_scalar_sum = 0.0

        for i, stage in enumerate(stages):
            # Compute cumulative downstream affine transform from stage i's output to final reference
            downstream_A = np.eye(2, dtype=np.float64)
            for j in range(i + 1, K):
                # stage j's 2x2 affine part
                downstream_A = stages[j].transform_matrix[:2, :2] @ downstream_A

            # Stage covariance
            if stage.uncertainty_cov is not None:
                stage_cov = stage.uncertainty_cov.astype(np.float64)
            else:
                var_i = stage.residual_rmse ** 2
                stage_cov = np.eye(2, dtype=np.float64) * var_i

            # Propagate stage covariance downstream
            propagated_cov = downstream_A @ stage_cov @ downstream_A.T
            composed_cov += propagated_cov

            # Scalar propagation: effective scale factor downstream
            s_downstream = float(np.linalg.norm(downstream_A, 2))  # spectral norm / maximum singular value
            variance_scalar_sum += (s_downstream * stage.residual_rmse) ** 2

        effective_uncertainty = math.sqrt(variance_scalar_sum)

        return CascadeResult(
            stages=stages,
            composed_transform=composed_H,
            effective_uncertainty_px=effective_uncertainty,
            composed_covariance=composed_cov,
            status=self.designation,
            notes=notes,
        )
