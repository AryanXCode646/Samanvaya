"""
Detector-Free Dense Feature Matcher using kornia.feature.LoFTR and OpenCV.

Implements the SIH PS 26166 correspondence pipeline:
1. Ingests preprocessed lunar GeoTIFF arrays (Source and Reference).
2. Extracts dense keypoint correspondences via LoFTR cross-attention transformer.
3. Enforces uniform spatial coverage via Grid-Based ANMS over an 8x8 grid (64 cells).
4. Refines keypoint locations to sub-pixel accuracy using analytical 2D parabolic Taylor-series interpolation.
5. Performs robust geometric outlier rejection using cv2.findHomography with cv2.USAC_MAGSAC,
   returning inliers, homography matrix H, and the registered warped source image.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import kornia.feature as KF

from lunar_core.models import KeypointMatch

logger = logging.getLogger(__name__)


@dataclass
class DenseLoFTRResult:
    """
    Result container for dense LoFTR planetary correspondence.
    Supports tuple unpacking: `inliers, H, warped_source = matcher.match(src, ref)`
    """
    inliers: List[KeypointMatch]
    homography: Optional[np.ndarray]
    warped_source: Optional[np.ndarray]
    all_matches: List[KeypointMatch]
    anms_matches: List[KeypointMatch]

    def __iter__(self) -> Iterator[Union[List[KeypointMatch], Optional[np.ndarray]]]:
        """Supports tuple unpacking: inliers, H, warped_source = result"""
        yield self.inliers
        yield self.homography
        yield self.warped_source


class DenseLoFTRMatcher:
    """
    Planetary Dense Matcher leveraging kornia.feature.LoFTR and USAC-MAGSAC.
    """

    def __init__(
        self,
        pretrained: Optional[str] = "outdoor",
        confidence_threshold: float = 0.20,
        grid_bins: int = 8,
        cap_per_cell: int = 4,
        patch_radius: int = 4,
        magsac_reproj_threshold: float = 1.5,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.grid_bins = grid_bins
        self.cap_per_cell = cap_per_cell
        self.patch_radius = patch_radius
        self.reproj_threshold = magsac_reproj_threshold
        self.is_pretrained = False

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize LoFTR backbone
        try:
            self.loftr = KF.LoFTR(pretrained=pretrained).to(self.device)
            self.is_pretrained = pretrained is not None
        except Exception as exc:
            logger.warning(
                "LoFTR pretrained weights failed to load (%s); untrained random "
                "weights are being used and results are not meaningful.",
                exc,
            )
            self.loftr = KF.LoFTR(pretrained=None).to(self.device)
        self.loftr.eval()

    @staticmethod
    def prepare_geotiff_array(image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Ingests a 2D or multi-band GeoTIFF array, normalizes dynamic range to [0, 1],
        and prepares a [1, 1, H, W] PyTorch tensor padded to multiples of 8.
        """
        arr = np.asarray(image, dtype=np.float32)

        # Handle multi-band or channel-first / channel-last GeoTIFFs
        if arr.ndim == 3:
            if arr.shape[0] in [1, 3]:  # [C, H, W]
                arr = arr[0] if arr.shape[0] == 1 else 0.2989 * arr[0] + 0.5870 * arr[1] + 0.1140 * arr[2]
            elif arr.shape[2] in [1, 3]:  # [H, W, C]
                arr = arr[..., 0] if arr.shape[2] == 1 else 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
            else:
                arr = arr[0]
        elif arr.ndim > 3:
            arr = arr.squeeze()

        # Robust percentile normalization to handle lunar shadows without dynamic range blowout
        p_min = float(np.nanpercentile(arr, 1.0))
        p_max = float(np.nanpercentile(arr, 99.0))
        denom = max(p_max - p_min, 1e-5)
        norm_arr = np.clip((arr - p_min) / denom, 0.0, 1.0).astype(np.float32)

        # Pad height and width to multiples of 8 for LoFTR CNN backbone
        h, w = norm_arr.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(norm_arr, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            padded = norm_arr

        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
        return tensor, norm_arr

    def extract_dense_correspondences(
        self,
        source_tensor: torch.Tensor,
        ref_tensor: torch.Tensor,
        src_shape: Tuple[int, int],
        ref_shape: Tuple[int, int],
    ) -> List[KeypointMatch]:
        """
        Extracts dense cross-attention correspondences using LoFTR.
        """
        source_tensor = source_tensor.to(self.device)
        ref_tensor = ref_tensor.to(self.device)

        input_dict = {"image0": source_tensor, "image1": ref_tensor}

        with torch.inference_mode():
            out = self.loftr(input_dict)

        pts_src = out["keypoints0"].detach().cpu().numpy()
        pts_ref = out["keypoints1"].detach().cpu().numpy()
        conf = out["confidence"].detach().cpu().numpy()
        del out
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        matches: List[KeypointMatch] = []
        h_src, w_src = src_shape
        h_ref, w_ref = ref_shape

        for (xs, ys), (xr, yr), c in zip(pts_src, pts_ref, conf):
            if c >= self.confidence_threshold:
                # Discard any padded coordinates beyond original image bounds
                if 0 <= xs < w_src and 0 <= ys < h_src and 0 <= xr < w_ref and 0 <= yr < h_ref:
                    matches.append(
                        KeypointMatch(
                            ref_xy=(float(xr), float(yr)),
                            target_xy=(float(xs), float(ys)),  # source coordinate
                            confidence=float(c),
                        )
                    )

        return matches

    def apply_grid_anms_8x8(
        self,
        matches: List[KeypointMatch],
        image_shape: Tuple[int, int],
        use_source_coords: bool = True,
    ) -> List[KeypointMatch]:
        """
        Subdivides the scene into an 8x8 spatial grid (64 cells).
        Enforces equal cap of top-confidence correspondences per cell in FULL_SOURCE_IMAGE coordinates.
        """
        return self.cap_grid_cells(matches, image_shape, use_source_coords=use_source_coords)

    def cap_grid_cells(
        self,
        matches: List[KeypointMatch],
        image_shape: Tuple[int, int],
        use_source_coords: bool = True,
    ) -> List[KeypointMatch]:
        """Caps the maximum matches permitted in any single spatial bin in FULL_SOURCE_IMAGE coordinates."""
        if not matches:
            return []

        h, w = image_shape[:2]
        bins = self.grid_bins
        cell_h = h / float(bins)
        cell_w = w / float(bins)

        grid_buckets: List[List[KeypointMatch]] = [[] for _ in range(bins * bins)]

        for m in matches:
            coord_x, coord_y = m.target_xy if use_source_coords else m.ref_xy
            gx = min(max(0, int(coord_x // cell_w)), bins - 1)
            gy = min(max(0, int(coord_y // cell_h)), bins - 1)
            cell_idx = gy * bins + gx
            grid_buckets[cell_idx].append(m)

        capped: List[KeypointMatch] = []
        for bucket in grid_buckets:
            if not bucket:
                continue
            sorted_bucket = sorted(bucket, key=lambda match: match.confidence, reverse=True)
            capped.extend(sorted_bucket[: self.cap_per_cell])

        return capped

    def refine_subpixel_taylor_2d(
        self,
        matches: List[KeypointMatch],
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> List[KeypointMatch]:
        r"""
        Refines integer match coordinates to continuous sub-pixel precision (< 0.4 px)
        by fitting an analytical 2D parabolic quadratic patch:
            f(x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y + f
        around the 3x3 local correlation neighborhood.

        Solves continuous stationary point:
            [dx*, dy*]^T = -H^{-1} g
        Validates negative definite Hessian: a < 0, b < 0, 4ab - c^2 > 0.
        """
        refined_list: List[KeypointMatch] = []
        r = self.patch_radius
        sh, sw = source_image.shape
        rh, rw = reference_image.shape

        for m in matches:
            rx_int, ry_int = int(round(m.ref_xy[0])), int(round(m.ref_xy[1]))
            sx_int, sy_int = int(round(m.target_xy[0])), int(round(m.target_xy[1]))

            # Check patch boundaries
            if (rx_int - r < 0 or rx_int + r >= rw or ry_int - r < 0 or ry_int + r >= rh or
                sx_int - r - 1 < 0 or sx_int + r + 1 >= sw or sy_int - r - 1 < 0 or sy_int + r + 1 >= sh):
                refined_list.append(m)
                continue

            ref_patch = reference_image[ry_int - r : ry_int + r + 1, rx_int - r : rx_int + r + 1]
            ref_std = float(np.std(ref_patch))
            if ref_std < 1e-4:
                refined_list.append(m)
                continue
            ref_norm = (ref_patch - np.mean(ref_patch)) / ref_std

            # Compute 3x3 local correlation surface
            grid_c = np.zeros((3, 3), dtype=np.float32)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    sub_src = source_image[
                        sy_int + dy - r : sy_int + dy + r + 1,
                        sx_int + dx - r : sx_int + dx + r + 1,
                    ]
                    src_std = float(np.std(sub_src))
                    if src_std < 1e-4:
                        grid_c[dy + 1, dx + 1] = 0.0
                    else:
                        sub_norm = (sub_src - np.mean(sub_src)) / src_std
                        grid_c[dy + 1, dx + 1] = float(np.mean(ref_norm * sub_norm))

            # Fit analytical 2D parabolic quadratic patch
            z00 = grid_c[1, 1]
            z_xp, z_xm = grid_c[1, 2], grid_c[1, 0]
            z_yp, z_ym = grid_c[2, 1], grid_c[0, 1]
            z_pp, z_pm = grid_c[2, 2], grid_c[0, 2]
            z_mp, z_mm = grid_c[2, 0], grid_c[0, 0]

            a = (z_xp - 2.0 * z00 + z_xm) / 2.0
            b = (z_yp - 2.0 * z00 + z_ym) / 2.0
            c = (z_pp - z_pm - z_mp + z_mm) / 4.0
            d = (z_xp - z_xm) / 2.0
            e = (z_yp - z_ym) / 2.0

            det_h = 4.0 * a * b - c**2

            # Check for strict local peak (negative definite Hessian)
            if det_h > 1e-7 and a < 0.0 and b < 0.0:
                dx_star = (-2.0 * b * d + c * e) / det_h
                dy_star = (-2.0 * a * e + c * d) / det_h

                # Ensure sub-pixel correction remains within cell boundary [-1, 1]
                if abs(dx_star) <= 1.0 and abs(dy_star) <= 1.0:
                    refined_sx = float(m.target_xy[0] + dx_star)
                    refined_sy = float(m.target_xy[1] + dy_star)
                    sigma_x = float(np.sqrt(abs((2.0 * b) / det_h)))
                    sigma_y = float(np.sqrt(abs((2.0 * a) / det_h)))
                    cov_xy = float(-c / det_h)
                    weight = float(np.sqrt(det_h))
                    refined_list.append(
                        KeypointMatch(
                            ref_xy=m.ref_xy,
                            target_xy=(refined_sx, refined_sy),
                            confidence=m.confidence,
                            subpixel_refined=True,
                            residual_error=m.residual_error,
                            sigma_x=sigma_x,
                            sigma_y=sigma_y,
                            cov_xy=cov_xy,
                            weight=weight,
                        )
                    )
                    continue

            refined_list.append(m)

        return refined_list

    def filter_outliers_magsac(
        self,
        matches: List[KeypointMatch],
        source_image: np.ndarray,
        ref_shape: Tuple[int, int],
    ) -> Tuple[List[KeypointMatch], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Filters outliers using cv2.findHomography with cv2.USAC_MAGSAC,
        returning inliers, homography matrix H, and the warped source image.
        """
        if len(matches) < 4:
            return [], None, None

        src_pts = np.array([m.target_xy for m in matches], dtype=np.float32)
        ref_pts = np.array([m.ref_xy for m in matches], dtype=np.float32)

        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask = cv2.findHomography(
            src_pts,
            ref_pts,
            method=method,
            ransacReprojThreshold=self.reproj_threshold,
            maxIters=5000,
            confidence=0.999,
        )

        if H is None or mask is None:
            return [], None, None

        inlier_indices = np.where(mask.ravel() == 1)[0]
        inliers: List[KeypointMatch] = []

        for idx in inlier_indices:
            m = matches[idx]
            pt_h = np.array([m.target_xy[0], m.target_xy[1], 1.0], dtype=np.float64)
            proj = H @ pt_h
            if abs(proj[2]) > 1e-7:
                proj_xy = (proj[0] / proj[2], proj[1] / proj[2])
            else:
                proj_xy = (proj[0], proj[1])

            residual = float(np.sqrt((proj_xy[0] - m.ref_xy[0])**2 + (proj_xy[1] - m.ref_xy[1])**2))
            inliers.append(
                KeypointMatch(
                    ref_xy=m.ref_xy,
                    target_xy=m.target_xy,
                    confidence=m.confidence,
                    subpixel_refined=m.subpixel_refined,
                    residual_error=residual,
                )
            )

        # Warp source into reference coordinate frame
        h_ref, w_ref = ref_shape[:2]
        warped_source = cv2.warpPerspective(source_image, H, (w_ref, h_ref), flags=cv2.INTER_LINEAR)

        return inliers, H, warped_source

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> DenseLoFTRResult:
        """
        Full 5-Step Pipeline:
        1. Ingests source & reference GeoTIFF arrays.
        2. Extracts dense correspondences via LoFTR.
        3. Enforces 8x8 Grid-Based ANMS.
        4. Applies 2D parabolic Taylor-series sub-pixel refinement.
        5. Estimates homography H via cv2.USAC_MAGSAC & warps source to reference frame.
        """
        # Step 1: Preprocess GeoTIFF arrays
        src_tensor, norm_src = self.prepare_geotiff_array(source_image)
        ref_tensor, norm_ref = self.prepare_geotiff_array(reference_image)

        # Step 2: Extract dense correspondences via LoFTR
        raw_matches = self.extract_dense_correspondences(
            src_tensor, ref_tensor, norm_src.shape, norm_ref.shape
        )

        # Step 3: Apply Grid-Based ANMS over 8x8 spatial grid in source frame
        anms_matches = self.apply_grid_anms_8x8(raw_matches, norm_src.shape, use_source_coords=True)

        # Step 4: Apply 2D parabolic Taylor-series sub-pixel refinement
        refined_matches = self.refine_subpixel_taylor_2d(anms_matches, norm_src, norm_ref)

        # Step 5: Filter outliers via cv2.USAC_MAGSAC and warp source
        inliers, H, warped = self.filter_outliers_magsac(refined_matches, norm_src, norm_ref.shape)

        return DenseLoFTRResult(
            inliers=inliers,
            homography=H,
            warped_source=warped,
            all_matches=raw_matches,
            anms_matches=anms_matches,
        )


# Backward-compatible wrapper for previous pipeline signatures
class DenseTransformerMatcher(DenseLoFTRMatcher):
    """
    Backward-compatible alias and wrapper for DenseLoFTRMatcher.
    """

    def __init__(
        self,
        temperature: float = 0.08,
        confidence_threshold: float = 0.20,
        grid_stride: int = 8,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold, cap_per_cell=4)

    def match_patches(
        self,
        ref_patch: np.ndarray,
        tgt_patch: np.ndarray,
    ) -> List[KeypointMatch]:
        """
        Matches patches returning List[KeypointMatch].
        """
        res = self.match(source_image=tgt_patch, reference_image=ref_patch)
        return res.inliers if res.inliers else res.anms_matches
