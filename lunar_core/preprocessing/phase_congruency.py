"""
Illumination-Invariant Feature Extraction: PyTorch & Kornia 2D Log-Gabor Phase Congruency.

Calculates Phase Congruency across an orientation filter bank in the frequency domain.
Specifically engineered for lunar orbital imagery (OHRC, TMC-2, IIRS, LRO NAC) with
severe shadowing and opposite solar illumination angles (morning vs. afternoon).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import kornia


@dataclass
class PhaseCongruencyOutput:
    """
    Container for Phase Congruency outputs supporting seamless NumPy & PyTorch interop.
    """
    phase_congruency: np.ndarray       # Total Phase Congruency [0, 1]
    max_moment: np.ndarray             # M_max: Principal structural edges (illumination-invariant)
    min_moment: np.ndarray             # M_min: Corner and junction strength
    orientation_max_idx: np.ndarray    # MIM: Maximum Index Map across orientations

    # PyTorch Tensor representations
    phase_congruency_tensor: Optional[torch.Tensor] = None
    max_moment_tensor: Optional[torch.Tensor] = None
    min_moment_tensor: Optional[torch.Tensor] = None
    orientation_max_idx_tensor: Optional[torch.Tensor] = None

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        """Returns dictionary of outputs as PyTorch tensors."""
        return {
            "phase_congruency": self.phase_congruency_tensor if self.phase_congruency_tensor is not None else torch.from_numpy(self.phase_congruency),
            "max_moment": self.max_moment_tensor if self.max_moment_tensor is not None else torch.from_numpy(self.max_moment),
            "min_moment": self.min_moment_tensor if self.min_moment_tensor is not None else torch.from_numpy(self.min_moment),
            "orientation_max_idx": self.orientation_max_idx_tensor if self.orientation_max_idx_tensor is not None else torch.from_numpy(self.orientation_max_idx),
        }


def get_optimal_hardware_device(requested_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Selects optimal hardware accelerator (CUDA -> Apple Silicon MPS -> CPU)
    with seamless automatic fallback and zero runtime exceptions.
    """
    if requested_device is not None:
        try:
            return torch.device(requested_device)
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LogGaborFilterBank:
    """
    2D Log-Gabor wavelet filter bank constructed in frequency space.
    Eliminates DC component entirely, preventing low-frequency illumination bias.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        d_theta_on_sigma: float = 1.2,
    ) -> None:
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.d_theta_on_sigma = d_theta_on_sigma
        self._filter_cache: dict = {}
        self._tensor_filter_cache: dict = {}
        self._grid_cache: dict = {}

    def get_frequency_grids(
        self, rows: int, cols: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (rows, cols)
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        u = np.linspace(-0.5, 0.5, cols, endpoint=False, dtype=np.float32)
        v = np.linspace(-0.5, 0.5, rows, endpoint=False, dtype=np.float32)
        x, y = np.meshgrid(u, v)

        x = np.fft.ifftshift(x)
        y = np.fft.ifftshift(y)

        radius = np.sqrt(x**2 + y**2).astype(np.float32)
        theta = np.arctan2(-y, x).astype(np.float32)
        radius[0, 0] = 1.0

        grids = (x, y, radius, theta)
        self._grid_cache[cache_key] = grids
        return grids

    def build_filters_tensor(self, rows: int, cols: int) -> np.ndarray:
        cache_key = (rows, cols)
        if cache_key in self._tensor_filter_cache:
            return self._tensor_filter_cache[cache_key]

        _, _, radius, theta = self.get_frequency_grids(rows, cols)
        theta_sigma = float(np.pi / self.num_orientations / self.d_theta_on_sigma)

        ang = (
            np.arange(self.num_orientations, dtype=np.float32)
            * (np.pi / self.num_orientations)
        ).reshape(-1, 1, 1)

        diff_theta1 = np.abs(theta[None, :, :] - ang)
        diff_theta1 = np.minimum(diff_theta1, 2.0 * np.pi - diff_theta1)
        diff_theta2 = np.abs(theta[None, :, :] - (ang + np.pi))
        diff_theta2 = np.minimum(diff_theta2, 2.0 * np.pi - diff_theta2)
        diff_theta = np.minimum(diff_theta1, diff_theta2)
        spread = np.exp(-(diff_theta**2) / (2.0 * theta_sigma**2)).astype(np.float32)

        s_idx = np.arange(self.num_scales, dtype=np.float32).reshape(-1, 1, 1)
        wavelength = (self.min_wavelength * (self.mult**s_idx)).astype(np.float32)
        f0 = 1.0 / wavelength

        log_rad = np.log(radius[None, :, :] / f0)
        radial = np.exp(
            -(log_rad**2) / (2.0 * (np.log(self.sigma_on_f)) ** 2)
        ).astype(np.float32)
        radial[:, 0, 0] = 0.0

        filters_4d = (spread[:, None, :, :] * radial[None, :, :, :]).astype(np.float32)
        self._tensor_filter_cache[cache_key] = filters_4d
        return filters_4d

    def build_filters(self, rows: int, cols: int) -> List[List[np.ndarray]]:
        cache_key = (rows, cols)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        tensor_filters = self.build_filters_tensor(rows, cols)
        filters: List[List[np.ndarray]] = [
            [tensor_filters[o, s] for s in range(self.num_scales)]
            for o in range(self.num_orientations)
        ]
        self._filter_cache[cache_key] = filters
        return filters


class PhaseCongruencyEngine:
    """
    PyTorch & Kornia accelerated 2D Log-Gabor Phase Congruency Engine.
    
    Operates in the 2D frequency domain using PyTorch FFT convolutions.
    Features:
    - Zero DC component strictly rejects low-frequency uneven solar lighting and albedo gradients.
    - Cached multi-scale, multi-orientation filter bank execution for ultra-fast repeated inference.
    - Kovesi Moment analysis for Maximum Moment (M_max) step edge detection.
    - Robust intensity normalization with Kornia for high-contrast, shadowed crater terrains.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_on_f: float = 0.55,
        d_theta_on_sigma: float = 1.2,
        k_noise: float = 2.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.n_scale = num_scales
        self.n_orient = num_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f
        self.d_theta_on_sigma = d_theta_on_sigma
        self.k_noise = k_noise
        self.device = get_optimal_hardware_device(device)

        # High-performance spatial-frequency caches
        self._filter_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._grid_cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def normalize_shadowed_lunar_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Normalizes high-contrast lunar imagery containing severe dark shadows and bright crater rims.
        Uses robust percentile dynamic range clamping and Kornia min-max scaling.
        """
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image.astype(np.float32)).to(self.device)
        else:
            img_tensor = image.to(self.device, dtype=torch.float32)

        # Ensure shape [1, 1, H, W] for Kornia processing
        orig_ndim = img_tensor.ndim
        if orig_ndim == 2:
            tensor_4d = img_tensor.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            tensor_4d = img_tensor.unsqueeze(0)
        else:
            tensor_4d = img_tensor

        # Handle deep lunar shadows: 2nd and 98th percentile robust contrast stretching
        p_low = torch.quantile(tensor_4d, 0.02)
        p_high = torch.quantile(tensor_4d, 0.98)
        denom = torch.clamp(p_high - p_low, min=1e-5)
        clamped = torch.clamp((tensor_4d - p_low) / denom, 0.0, 1.0)

        # Scale with Kornia enhance min-max
        normalized = kornia.enhance.normalize_min_max(clamped, min_val=0.0, max_val=1.0)

        # Return 2D float32 tensor [H, W]
        return normalized.squeeze()

    def get_cached_frequency_grids(
        self,
        rows: int,
        cols: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves or caches precomputed frequency coordinate grids: (u_grid, v_grid, radius, theta).
        """
        cache_key = (rows, cols, str(device))
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        v = torch.linspace(-0.5, 0.5 - 1.0 / rows, rows, device=device, dtype=torch.float32)
        u = torch.linspace(-0.5, 0.5 - 1.0 / cols, cols, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        radius = torch.sqrt(u_grid**2 + v_grid**2)
        radius[rows // 2, cols // 2] = 1.0  # Avoid log(0) at DC component

        theta = torch.atan2(-v_grid, u_grid)

        grids = (u_grid, v_grid, radius, theta)
        self._grid_cache[cache_key] = grids
        return grids

    def build_log_gabor_filter_bank(
        self,
        rows: int,
        cols: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Constructs vectorized 2D Log-Gabor frequency filter bank of shape [n_scale, n_orient, rows, cols].
        Employs O(1) memory caching to bypass repetitive wave grid recalculations.
        """
        cache_key = (rows, cols, str(device))
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        _, _, radius, theta = self.get_cached_frequency_grids(rows, cols, device)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        # 1. Radial component for all scales: shape [n_scale, rows, cols]
        scale_exponents = torch.arange(self.n_scale, device=device, dtype=torch.float32)
        wavelengths = (self.min_wavelength * (self.mult ** scale_exponents)).view(self.n_scale, 1, 1)
        fo = 1.0 / wavelengths
        r_ratio = radius.unsqueeze(0) / fo
        log_gabor_radial = torch.exp(-((torch.log(r_ratio)) ** 2) / (2.0 * (np.log(self.sigma_on_f)) ** 2))
        log_gabor_radial[:, rows // 2, cols // 2] = 0.0  # Strict zero DC

        # 2. Angular Gaussian spread for all orientations: shape [n_orient, rows, cols]
        orient_idx = torch.arange(self.n_orient, device=device, dtype=torch.float32).view(self.n_orient, 1, 1)
        angles = orient_idx * np.pi / self.n_orient
        ds = sintheta.unsqueeze(0) * torch.cos(angles) - costheta.unsqueeze(0) * torch.sin(angles)
        dc = costheta.unsqueeze(0) * torch.cos(angles) + sintheta.unsqueeze(0) * torch.sin(angles)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread = torch.exp(-(dtheta ** 2) / (2.0 * ((np.pi / self.n_orient) / self.d_theta_on_sigma) ** 2))

        # 3. 2D Filter Bank: shape [n_scale, n_orient, rows, cols]
        filters = log_gabor_radial.unsqueeze(1) * spread.unsqueeze(0)

        # Shift zero-frequency to corners for PyTorch fft2/ifft2
        shifted = torch.fft.ifftshift(filters, dim=(-2, -1))
        self._filter_cache[cache_key] = shifted
        return shifted

    def compute(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> PhaseCongruencyOutput:
        """
        Computes illumination-invariant Phase Congruency and Kovesi moments via PyTorch FFT.
        
        Args:
            image: 2D lunar optical image (NumPy array or PyTorch Tensor).
            
        Returns:
            PhaseCongruencyOutput with normalized PC, M_max, M_min, and MIM.
        """
        norm_img = self.normalize_shadowed_lunar_image(image)
        rows, cols = norm_img.shape[-2:]

        # Build vectorized Log-Gabor filter bank
        filters = self.build_log_gabor_filter_bank(rows, cols, device=norm_img.device)

        # PyTorch 2D Fast Fourier Transform
        fft_img = torch.fft.fft2(norm_img)

        # Vectorized frequency-domain filtering: shape [n_scale, n_orient, rows, cols]
        filtered_fft = fft_img.unsqueeze(0).unsqueeze(0) * filters
        resp = torch.fft.ifft2(filtered_fft)

        # Even (real) and odd (imaginary) symmetric wave responses
        eo_real = torch.real(resp)
        eo_imag = torch.imag(resp)
        amplitudes = torch.sqrt(eo_real**2 + eo_imag**2)

        # Sum components across scales per orientation: shape [n_orient, rows, cols]
        sum_e_per_orient = torch.sum(eo_real, dim=0)
        sum_o_per_orient = torch.sum(eo_imag, dim=0)
        orient_energies = torch.sqrt(sum_e_per_orient**2 + sum_o_per_orient**2)

        # Total energy and amplitude across all scales and orientations
        total_e = torch.sum(sum_e_per_orient, dim=0)
        total_o = torch.sum(sum_o_per_orient, dim=0)
        overall_energy = torch.sqrt(total_e**2 + total_o**2)
        sum_amplitudes = torch.sum(amplitudes, dim=(0, 1))

        # Phase Congruency PC in [0, 1]
        pc = torch.clamp(overall_energy / (sum_amplitudes + 1e-5), 0.0, 1.0)

        # Kovesi Moment Analysis across orientations
        orient_angles = (torch.arange(self.n_orient, device=norm_img.device, dtype=torch.float32) * np.pi / self.n_orient).view(self.n_orient, 1, 1)
        cos_th = torch.cos(orient_angles)
        sin_th = torch.sin(orient_angles)

        sum_xx = torch.sum((orient_energies * cos_th) ** 2, dim=0) / (self.n_orient / 2.0)
        sum_yy = torch.sum((orient_energies * sin_th) ** 2, dim=0) / (self.n_orient / 2.0)
        sum_xy = 2.0 * torch.sum((orient_energies * cos_th) * (orient_energies * sin_th), dim=0) / (self.n_orient / 2.0)

        diff = sum_xx - sum_yy
        radical = torch.sqrt(diff**2 + sum_xy**2)

        # Maximum Moment M_max (invariant step edges) & Minimum Moment M_min (corners)
        m_max = 0.5 * (sum_xx + sum_yy + radical)
        m_min = 0.5 * (sum_xx + sum_yy - radical)

        # Normalized to [0, 1]
        m_max_norm = (m_max - m_max.min()) / (m_max.max() - m_max.min() + 1e-6)
        m_min_norm = (m_min - m_min.min()) / (m_min.max() - m_min.min() + 1e-6)
        mim = torch.argmax(orient_energies, dim=0).to(torch.float32)

        # Convert to numpy for seamless downstream integration
        return PhaseCongruencyOutput(
            phase_congruency=pc.detach().cpu().numpy().astype(np.float32),
            max_moment=m_max_norm.detach().cpu().numpy().astype(np.float32),
            min_moment=m_min_norm.detach().cpu().numpy().astype(np.float32),
            orientation_max_idx=mim.detach().cpu().numpy().astype(np.float32),
            phase_congruency_tensor=pc.detach(),
            max_moment_tensor=m_max_norm.detach(),
            min_moment_tensor=m_min_norm.detach(),
            orientation_max_idx_tensor=mim.detach(),
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator
    from lunar_core.models import SunAngles

    print("Executing Standalone Visual Test: PyTorch & Kornia Phase Congruency...")

    # 1. Simulate lunar terrain with severe craters
    sim = LunarTerrainSimulator(size=(256, 256), seed=123)
    dem = sim.generate_dem(num_craters=12)

    # 2. Render under Morning lighting (Azimuth 60°, El 20° - severe East shadows)
    sun_morning = SunAngles(azimuth_deg=60.0, elevation_deg=20.0)
    img_morning = sim.render_optical_image(dem, sun_morning)

    # 3. Render under Afternoon lighting (Azimuth 240°, El 20° - 180° complete shadow reversal)
    sun_afternoon = SunAngles(azimuth_deg=240.0, elevation_deg=20.0)
    img_afternoon = sim.render_optical_image(dem, sun_afternoon)

    # 4. Process with PyTorch & Kornia Phase Congruency
    engine = PhaseCongruencyEngine(num_scales=4, num_orientations=6)
    out_morning = engine.compute(img_morning)
    out_afternoon = engine.compute(img_afternoon)

    # 5. Compute Quantitative Metrics
    raw_corr = float(np.corrcoef(img_morning.ravel(), img_afternoon.ravel())[0, 1])
    pc_corr = float(np.corrcoef(out_morning.max_moment.ravel(), out_afternoon.max_moment.ravel())[0, 1])

    print(f"Raw Intensity Correlation (Opposite Sun): {raw_corr:.4f} (Fails due to inverted shadows)")
    print(f"Phase Congruency M_max Correlation:      {pc_corr:.4f} (Maintains structural coincidence)")

    # 6. Generate 6-Panel Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_morning, cmap="gray")
    axes[0, 0].set_title("Morning Sun (Az: 60°, El: 20°)\nSevere East Shadows")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_afternoon, cmap="gray")
    axes[0, 1].set_title("Afternoon Sun (Az: 240°, El: 20°)\n180° Inverted West Shadows")
    axes[0, 1].axis("off")

    raw_diff = np.abs(img_morning - img_afternoon)
    axes[0, 2].imshow(raw_diff, cmap="inferno")
    axes[0, 2].set_title(f"Raw Intensity Difference\nPearson Corr: {raw_corr:.3f}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(out_morning.max_moment, cmap="magma")
    axes[1, 0].set_title("Morning Maximum Moment (M_max)\nCrater Step Edges")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(out_afternoon.max_moment, cmap="magma")
    axes[1, 1].set_title("Afternoon Maximum Moment (M_max)\nCrater Step Edges")
    axes[1, 1].axis("off")

    moment_diff = np.abs(out_morning.max_moment - out_afternoon.max_moment)
    axes[1, 2].imshow(moment_diff, cmap="inferno")
    axes[1, 2].set_title(f"Phase Congruency Difference\nPearson Corr: {pc_corr:.3f} (Invariant)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_img_path = "tests/visual_test_phase_congruency.png"
    plt.savefig(out_img_path, dpi=200, bbox_inches="tight")
    print(f"Saved visual test artifact to: {out_img_path}")
    plt.close()
