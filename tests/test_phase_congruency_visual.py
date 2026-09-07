"""
Standalone Visual Test for PyTorch & Kornia Phase Congruency under Simulated Inverted Lighting.
SIH PS 26166: Illumination-Invariant Planetary Step Edge Extraction.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from lunar_core.models import SunAngles
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator


def test_phase_congruency_pytorch_invariance_under_inverted_lighting():
    """
    Demonstrates that PyTorch & Kornia Phase Congruency extracts invariant crater rim
    step edges despite 180-degree shadow reversals and severe unilluminated shadows.
    """
    # 1. Simulate lunar terrain with distinct impact craters
    sim = LunarTerrainSimulator(size=(256, 256), seed=777)
    dem = sim.generate_dem(num_craters=12)

    # 2. Render high-contrast morning scene (Azimuth 60°, Elevation 20° - long East shadows)
    sun_morning = SunAngles(azimuth_deg=60.0, elevation_deg=20.0)
    img_morning = sim.render_optical_image(dem, sun_morning)

    # 3. Render high-contrast afternoon scene (Azimuth 240°, Elevation 20° - 180° complete shadow reversal)
    sun_afternoon = SunAngles(azimuth_deg=240.0, elevation_deg=20.0)
    img_afternoon = sim.render_optical_image(dem, sun_afternoon)

    # 4. Instantiate PyTorch Phase Congruency Engine
    engine = PhaseCongruencyEngine(num_scales=4, num_orientations=6, device="cpu")

    # Run on NumPy array and PyTorch Tensor to verify dual compatibility
    out_morning = engine.compute(img_morning)
    img_afternoon_tensor = torch.from_numpy(img_afternoon).float()
    out_afternoon = engine.compute(img_afternoon_tensor)

    # Verify tensor outputs
    tensors = out_morning.as_tensors()
    assert isinstance(tensors["max_moment"], torch.Tensor)
    assert tensors["max_moment"].shape == (256, 256)

    # 5. Quantitative Invariance Analysis
    raw_corr = float(np.corrcoef(img_morning.ravel(), img_afternoon.ravel())[0, 1])
    pc_corr = float(np.corrcoef(out_morning.max_moment.ravel(), out_afternoon.max_moment.ravel())[0, 1])

    print(f"\n[Visual Test Invariance Audit]")
    print(f"Raw Intensity Correlation (Opposite Sun): {raw_corr:.4f} (Fails due to inverted shadows)")
    print(f"Phase Congruency M_max Correlation:      {pc_corr:.4f} (Invariant physical step edges)")

    # Assert invariant structural correlation is substantially higher than raw intensity
    assert pc_corr > raw_corr
    assert pc_corr > 0.40, f"M_max correlation across 180 deg illumination reversal must be > 0.40, got {pc_corr:.4f}"

    # 6. Generate Standalone Visual Inspection Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top Row: Optical Images and Raw Intensity Difference
    axes[0, 0].imshow(img_morning, cmap="gray")
    axes[0, 0].set_title("Morning Sun (Az: 60°, El: 20°)\nSevere East Shadows", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_afternoon, cmap="gray")
    axes[0, 1].set_title("Afternoon Sun (Az: 240°, El: 20°)\n180° Inverted West Shadows", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    raw_diff = np.abs(img_morning - img_afternoon)
    im_raw_diff = axes[0, 2].imshow(raw_diff, cmap="inferno")
    axes[0, 2].set_title(f"Raw Pixel Difference\nPearson Corr: {raw_corr:.3f} (Severe Error)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im_raw_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom Row: PyTorch Phase Congruency Maximum Moments (M_max) and Moment Difference
    axes[1, 0].imshow(out_morning.max_moment, cmap="magma")
    axes[1, 0].set_title("Morning Maximum Moment (M_max)\nPyTorch & Kornia Frequency Edges", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(out_afternoon.max_moment, cmap="magma")
    axes[1, 1].set_title("Afternoon Maximum Moment (M_max)\nPyTorch & Kornia Frequency Edges", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    moment_diff = np.abs(out_morning.max_moment - out_afternoon.max_moment)
    im_moment_diff = axes[1, 2].imshow(moment_diff, cmap="inferno")
    axes[1, 2].set_title(f"M_max Difference Map\nPearson Corr: {pc_corr:.3f} (Illumination Invariant)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(im_moment_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle("PyTorch & Kornia Phase Congruency: 180° Solar Shadow Inversion Test", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save artifact to project tests/ and artifact directory
    test_out_path = Path("tests/visual_test_phase_congruency.png")
    test_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(test_out_path, dpi=180, bbox_inches="tight")

    artifact_dir = Path("/home/zx0/.gemini/antigravity-cli/brain/1e5280c1-00e9-46b4-b235-ddfc147649f0")
    if artifact_dir.exists():
        artifact_path = artifact_dir / "visual_test_phase_congruency.png"
        plt.savefig(artifact_path, dpi=180, bbox_inches="tight")
        print(f"Artifact successfully saved to {artifact_path}")

    plt.close()
    assert test_out_path.exists(), "Output visualization plot must exist on disk"


if __name__ == "__main__":
    test_phase_congruency_pytorch_invariance_under_inverted_lighting()
    print("Standalone visual test executed successfully!")
