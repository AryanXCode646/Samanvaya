"""
Planetary Photometric Reflectance Normalizer (Lommel-Seeliger & Hapke).
Supports 3D Digital Elevation Model (DEM) slope gradients and surface normal facet derivation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from lunar_core.models import SunAngles

logger = logging.getLogger("lunar_core.photometric")


class PhotometricNormalizer:
    """
    Normalizes optical lunar imagery against variable solar incidence, emission, and phase angles.
    Incorporates high-resolution DEM topographic gradients (dz/dx, dz/dy) to resolve steep crater walls (20°-45°).
    """

    def __init__(self, ref_incidence_deg: float = 30.0, ref_emission_deg: float = 0.0) -> None:
        self.ref_i = float(np.radians(ref_incidence_deg))
        self.ref_e = float(np.radians(ref_emission_deg))

    @staticmethod
    def lommel_seeliger(cos_i: np.ndarray, cos_e: np.ndarray) -> np.ndarray:
        """
        Lommel-Seeliger scattering law for particulate planetary regolith:
            R_LS(i, e) = cos(i) / (cos(i) + cos(e))
        """
        denom = np.maximum(cos_i + cos_e, 1e-6)
        return np.maximum(cos_i, 0.0) / denom

    @staticmethod
    def compute_surface_normals_from_dem(
        dem_data: np.ndarray,
        pixel_gsd: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes continuous 3D unit normal vectors [nx, ny, nz] from a 2D DEM array.
        Uses 3x3 Sobel operators scaled by local pixel GSD.

        Surface Normal formulation:
            dz/dx = Sobel_x(dem) / (8 * pixel_gsd)
            dz/dy = Sobel_y(dem) / (8 * pixel_gsd)
            n = [-dz/dx, -dz/dy, 1] / sqrt((dz/dx)^2 + (dz/dy)^2 + 1)

        Returns:
            Tuple of (nx, ny, nz) unit vector component arrays.
        """
        dem_f = dem_data.astype(np.float32)

        # Scale Sobel gradients by standard 8-weight kernel factor and spatial pixel GSD
        scale_factor = 8.0 * max(float(pixel_gsd), 1e-6)
        dz_dx = cv2.Sobel(dem_f, cv2.CV_32F, 1, 0, ksize=3) / scale_factor
        dz_dy = cv2.Sobel(dem_f, cv2.CV_32F, 0, 1, ksize=3) / scale_factor

        # Magnitude of the facet normal vector
        norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)

        nx = -dz_dx / norm
        ny = -dz_dy / norm
        nz = 1.0 / norm

        return nx, ny, nz

    def normalize(
        self,
        image: np.ndarray,
        sun_angles: SunAngles,
        dem_data: Optional[np.ndarray] = None,
        slope_gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pixel_gsd: float = 1.0,
        max_correction_factor: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects image intensities to standard reference lighting geometry.

        Args:
            image: 2D array of raw optical intensities.
            sun_angles: Solar incidence and azimuth angles.
            dem_data: Optional aligned 2D digital elevation model (DEM) array in meters.
            slope_gradients: Optional precomputed (dz/dx, dz/dy) spatial gradients.
            pixel_gsd: Ground Sampling Distance in meters per pixel.
            max_correction_factor: Upper bound on photometric multiplier to prevent rim burnout.

        Returns:
            Tuple of (corrected_image_float32, valid_illumination_mask).
        """
        img_f = image.astype(np.float32)
        h, w = img_f.shape
        sun_vec = sun_angles.sun_vector  # [sx, sy, sz] normalized

        # Nadir viewing vector v = [0, 0, 1]
        if dem_data is not None or slope_gradients is not None:
            # Topographic DEM slope derivation
            if dem_data is not None:
                # Resize DEM if slight resolution mismatch with optical frame
                if dem_data.shape != (h, w):
                    dem_f = cv2.resize(dem_data.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    dem_f = dem_data
                nx, ny, nz = self.compute_surface_normals_from_dem(dem_f, pixel_gsd)
            else:
                dz_dx, dz_dy = slope_gradients
                norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
                nx = -dz_dx / norm
                ny = -dz_dy / norm
                nz = 1.0 / norm

            # Continuous facet incidence angle: cos_i = n · s
            cos_i = nx * sun_vec[0] + ny * sun_vec[1] + nz * sun_vec[2]

            # Continuous facet emission angle: cos_e = n · v = nz
            cos_e = nz
        else:
            # Graceful Fallback: Planar horizontal surface normal n = [0, 0, 1]
            cos_i = np.full((h, w), np.maximum(sun_vec[2], 0.01), dtype=np.float32)
            cos_e = np.ones((h, w), dtype=np.float32)

        # Reference geometry reflectance
        ref_rls = self.lommel_seeliger(
            np.array(np.cos(self.ref_i), dtype=np.float32),
            np.array(np.cos(self.ref_e), dtype=np.float32),
        )

        # Local acquisition reflectance under topographic slopes
        safe_cos_i = np.maximum(cos_i, 0.005)
        safe_cos_e = np.maximum(cos_e, 0.005)
        acq_rls = self.lommel_seeliger(safe_cos_i, safe_cos_e)

        # Correction ratio: R_ref / R_acq
        # Bounded to prevent rim burnout on steep slopes facing away from the sun
        corr = np.where(acq_rls > 1e-4, ref_rls / np.maximum(acq_rls, 1e-4), 1.0)
        corr = np.clip(corr, 0.1, max_correction_factor)

        corrected = img_f * corr

        # Valid illumination mask (exclude cast shadows and extreme grazing facets)
        shadow_thresh = float(np.nanquantile(img_f, 0.05))
        valid_mask = (img_f > shadow_thresh) & (cos_i > 0.04)

        # Robust contrast normalization on illuminated regolith
        valid_pixels = corrected[valid_mask]
        if len(valid_pixels) > 0:
            p2 = float(np.nanpercentile(valid_pixels, 2.0))
            p98 = float(np.nanpercentile(valid_pixels, 98.0))
            if p98 > p2:
                corrected = np.clip((corrected - p2) / (p98 - p2), 0.0, 1.0)
            else:
                corrected = np.clip(corrected, 0.0, 1.0)
        else:
            corrected = np.clip(corrected, 0.0, 1.0)

        return corrected.astype(np.float32), valid_mask

    @staticmethod
    def minnaert(cos_i: np.ndarray, cos_e: np.ndarray, k: float = 0.8, eps: float = 1e-6) -> np.ndarray:
        """
        Minnaert photometric scattering model for rough planetary surfaces:
            R_Minnaert = (cos i)^k * (cos e)^(k - 1)
        where k in [0.5, 1.0] is the limb-darkening exponent (default k=0.8 for lunar regolith).
        """
        mu_0 = np.maximum(cos_i, eps)
        mu = np.maximum(cos_e, eps)
        return np.maximum((mu_0 ** k) * (mu ** (k - 1.0)), 0.0)

    def normalize_minnaert(
        self,
        image: np.ndarray,
        sun_angles: SunAngles,
        dem_data: Optional[np.ndarray] = None,
        slope_gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pixel_gsd: float = 1.0,
        k: float = 0.8,
        max_correction_factor: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects image intensities via Topographic Illumination Correction & Minnaert law.
        Suppresses shadow boundary step discontinuities across steep crater walls.
        """
        img_f = image.astype(np.float32)
        h, w = img_f.shape
        sun_vec = sun_angles.sun_vector

        if dem_data is not None or slope_gradients is not None:
            if dem_data is not None:
                if dem_data.shape != (h, w):
                    dem_f = cv2.resize(dem_data.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    dem_f = dem_data
                nx, ny, nz = self.compute_surface_normals_from_dem(dem_f, pixel_gsd)
            else:
                dz_dx, dz_dy = slope_gradients
                norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
                nx = -dz_dx / norm
                ny = -dz_dy / norm
                nz = 1.0 / norm
            cos_i = nx * sun_vec[0] + ny * sun_vec[1] + nz * sun_vec[2]
            cos_e = nz
        else:
            cos_i = np.full((h, w), np.maximum(sun_vec[2], 0.01), dtype=np.float32)
            cos_e = np.ones((h, w), dtype=np.float32)

        ref_cos_i = np.cos(self.ref_i)
        ref_cos_e = np.cos(self.ref_e)
        ref_rm = self.minnaert(np.array(ref_cos_i, dtype=np.float32), np.array(ref_cos_e, dtype=np.float32), k=k)

        safe_cos_i = np.maximum(cos_i, 0.005)
        safe_cos_e = np.maximum(cos_e, 0.005)
        acq_rm = self.minnaert(safe_cos_i, safe_cos_e, k=k)

        corr = np.where(acq_rm > 1e-5, ref_rm / np.maximum(acq_rm, 1e-5), 1.0)
        corr = np.clip(corr, 0.1, max_correction_factor)
        corrected = img_f * corr

        shadow_thresh = float(np.nanquantile(img_f, 0.05))
        valid_mask = (img_f > shadow_thresh) & (cos_i > 0.04)

        valid_pixels = corrected[valid_mask]
        if len(valid_pixels) > 0:
            p2 = float(np.nanpercentile(valid_pixels, 2.0))
            p98 = float(np.nanpercentile(valid_pixels, 98.0))
            if p98 > p2:
                corrected = np.clip((corrected - p2) / (p98 - p2), 0.0, 1.0)
            else:
                corrected = np.clip(corrected, 0.0, 1.0)
        else:
            corrected = np.clip(corrected, 0.0, 1.0)

        return corrected.astype(np.float32), valid_mask

    def normalize_with_provenance(
        self,
        image: np.ndarray,
        sun_angles: Optional[SunAngles],
        dem_data: Optional[np.ndarray] = None,
        slope_gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pixel_gsd: float = 1.0,
        model: str = "lommel_seeliger",
    ) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Normalize illumination with explicit metadata provenance matching Section 7."""
        if sun_angles is None:
            return image.astype(np.float32), np.ones(image.shape, dtype=bool), {
                "photometric_model": "none",
                "geometry_source": "none",
                "fallback": True,
                "reason": "solar_geometry_unavailable",
            }

        has_dem = dem_data is not None or slope_gradients is not None
        geom_source = "dem_slope_facets" if has_dem else "pds4_sun_angles_planar"

        if model.lower() == "minnaert":
            corr, mask = self.normalize_minnaert(
                image,
                sun_angles,
                dem_data=dem_data,
                slope_gradients=slope_gradients,
                pixel_gsd=pixel_gsd,
            )
            model_name = "Minnaert"
        else:
            corr, mask = self.normalize(
                image,
                sun_angles,
                dem_data=dem_data,
                slope_gradients=slope_gradients,
                pixel_gsd=pixel_gsd,
            )
            model_name = "Lommel-Seeliger"

        provenance = {
            "photometric_model": model_name,
            "geometry_source": geom_source,
            "sun_azimuth_deg": sun_angles.azimuth_deg,
            "sun_elevation_deg": sun_angles.elevation_deg,
            "fallback": not has_dem,
            "reason": None if has_dem else "planar_geometry_dem_unavailable",
        }
        return corr, mask, provenance

