"""
🌙 ISRO Chandrayaan-2 Planetary Image Registration Portal.
SIH PS 26166: Multi-Modal, Sun-Angle, and Scale-Invariant Lunar Correspondence.

Built with Streamlit, PyTorch, Kornia, and OpenCV.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union

# Ensure project root is in sys.path regardless of execution working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from lunar_core.models import SensorModality, SunAngles, KeypointMatch
from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.preprocessing.photometric import PhotometricNormalizer
from lunar_core.evaluation.metrics import EvaluationEngine, RegistrationEvaluationReport
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator


# Configure Streamlit Page
st.set_page_config(
    page_title="ISRO Chandrayaan-2 Lunar Alignment Portal",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌙 ISRO Chandrayaan-2 Lunar Image Registration Portal")
st.markdown("### SIH PS 26166: Multi-Modal (OHRC / TMC-2 / IIRS vs. LRO NAC), Sun-Angle & Scale-Invariant Alignment")

# -----------------------------------------------------------------------------
# Helper Functions: Image Loading, Sample Presets & Plotting
# -----------------------------------------------------------------------------

def get_sample_data_dir() -> Path:
    """Robustly locates lunar_core/assets/sample_data across local, CLI, and Docker environments."""
    local_path = Path(__file__).resolve().parent.parent / "assets" / "sample_data"
    if local_path.exists() and (local_path / "manifest.json").exists():
        return local_path
    cwd_path = Path.cwd() / "lunar_core" / "assets" / "sample_data"
    if cwd_path.exists() and (cwd_path / "manifest.json").exists():
        return cwd_path
    root_path = _PROJECT_ROOT / "lunar_core" / "assets" / "sample_data"
    if root_path.exists() and (root_path / "manifest.json").exists():
        return root_path
    return local_path


@st.cache_data(show_spinner=False)
def load_sample_manifest() -> dict:
    """Loads benchmark preset metadata catalog."""
    manifest_file = get_sample_data_dir() / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@st.cache_data(show_spinner=False)
def load_geotiff_file(file_path: Union[str, Path]) -> np.ndarray:
    """Safely loads a cached GeoTIFF from disk as normalized float32 [0.0, 1.0]."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {path}")
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            p_low = float(np.nanpercentile(arr, 1.0))
            p_high = float(np.nanpercentile(arr, 99.0))
            denom = max(p_high - p_low, 1e-5)
            return np.clip((arr - p_low) / denom, 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fallback to OpenCV
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image file {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_f = img.astype(np.float32)
        p_low, p_high = np.percentile(img_f, 1.0), np.percentile(img_f, 99.0)
        denom = max(float(p_high - p_low), 1e-5)
        return np.clip((img_f - p_low) / denom, 0.0, 1.0).astype(np.float32)


def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Safely loads an uploaded GeoTIFF, TIFF, PNG, or JPEG file as a 2D float32 array."""
    bytes_data = uploaded_file.getvalue()
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(bytes_data) as memfile:
            with memfile.open() as src:
                arr = src.read(1).astype(np.float32)
                # Filter out extreme NaN/NoData values
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                p_low, p_high = np.nanpercentile(arr, 1.0), np.nanpercentile(arr, 99.0)
                denom = max(float(p_high - p_low), 1e-5)
                return np.clip((arr - p_low) / denom, 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fallback to OpenCV
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Unsupported image format")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        p_low, p_high = np.percentile(img, 1.0), np.percentile(img, 99.0)
        denom = max(float(p_high - p_low), 1e-5)
        return np.clip((img - p_low) / denom, 0.0, 1.0).astype(np.float32)


def render_tie_point_correspondences(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    matches: list[KeypointMatch],
    max_display: int = 50,
) -> plt.Figure:
    """
    Renders an interactive side-by-side tie-point correspondence plot
    color-coded by match confidence score.
    """
    h_src, w_src = source_img.shape
    h_ref, w_ref = reference_img.shape

    canvas_h = max(h_src, h_ref)
    canvas_w = w_src + w_ref
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas[:h_src, :w_src] = source_img
    canvas[:h_ref, w_src:canvas_w] = reference_img

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(canvas, cmap="gray")

    if matches:
        # Display top matches sorted by confidence
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)[:max_display]
        confidences = np.array([m.confidence for m in sorted_matches])

        cmap = plt.cm.turbo
        norm = plt.Normalize(vmin=max(0.0, float(np.min(confidences))), vmax=1.0)

        for m in sorted_matches:
            xs, ys = m.target_xy  # Source coordinate
            xr, yr = m.ref_xy     # Reference coordinate
            color = cmap(norm(m.confidence))

            # Connecting correspondence line
            ax.plot([xs, xr + w_src], [ys, yr], color=color, alpha=0.75, linewidth=1.2)
            # Source marker (Left)
            ax.scatter(xs, ys, color=color, s=28, edgecolors="black", linewidths=0.5)
            # Reference marker (Right)
            ax.scatter(xr + w_src, yr, color=color, s=28, edgecolors="black", linewidths=0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.08, shrink=0.6)
        cbar.set_label("Match Confidence Score (LoFTR Cross-Attention)", fontsize=10)

    # Annotate Source & Reference Regions
    ax.text(w_src / 2.0, -12, "Source Image (OHRC / TMC-2)", ha="center", va="bottom", fontsize=12, fontweight="bold", color="yellow")
    ax.text(w_src + w_ref / 2.0, -12, "Reference Image (LRO NAC / Base)", ha="center", va="bottom", fontsize=12, fontweight="bold", color="cyan")
    ax.axvline(w_src, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axis("off")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Sidebar Configuration & Mission Evaluation Benchmarks
# -----------------------------------------------------------------------------

st.sidebar.header("🎯 Mission Evaluation Benchmark")
sample_manifest = load_sample_manifest()
benchmarks = sample_manifest.get("benchmarks", {})

benchmark_options = {
    "Scenario A: Chandrayaan-2 OHRC vs LRO NAC (Apollo 11 Landing Site, sub-meter)": "scenario_a",
    "Scenario B: TMC-2 Nadir vs TMC-2 Fore (Stereo baseline pair, 5 m/px)": "scenario_b",
    "Scenario C: Extreme Solar Lighting Disparity (Low Sun 12° vs High Sun 65°)": "scenario_c",
    "Synthetic Lunar Crater Simulation": "synthetic_sim",
    "Custom GeoTIFF Upload": "custom_upload",
}

selected_benchmark = st.sidebar.selectbox(
    "Select Mission Evaluation Benchmark",
    list(benchmark_options.keys()),
    index=0,
)
selected_key = benchmark_options[selected_benchmark]

# Automatic modality syncing based on selected preset
default_src_mod = SensorModality.OHRC.value
default_ref_mod = SensorModality.LRO_NAC.value

if selected_key == "scenario_b":
    default_src_mod = SensorModality.TMC2.value
    default_ref_mod = SensorModality.TMC2.value
elif selected_key == "scenario_c":
    default_src_mod = SensorModality.OHRC.value
    default_ref_mod = SensorModality.LRO_NAC.value

st.sidebar.markdown("---")
st.sidebar.header("🛰️ Planetary Sensor Configuration")
src_idx = [SensorModality.OHRC.value, SensorModality.TMC2.value, SensorModality.IIRS.value].index(default_src_mod)
ref_idx = [SensorModality.LRO_NAC.value, SensorModality.TMC2.value].index(default_ref_mod)

source_modality = st.sidebar.selectbox("Source Modality", [SensorModality.OHRC.value, SensorModality.TMC2.value, SensorModality.IIRS.value], index=src_idx)
ref_modality = st.sidebar.selectbox("Reference Modality", [SensorModality.LRO_NAC.value, SensorModality.TMC2.value], index=ref_idx)

uploaded_src = None
uploaded_ref = None

if selected_key == "custom_upload":
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Custom GeoTIFF Rasters")
    uploaded_src = st.sidebar.file_uploader("Upload Source GeoTIFF (OHRC / TMC-2)", type=["tif", "tiff", "geotiff", "png", "jpg"])
    uploaded_ref = st.sidebar.file_uploader("Upload Reference GeoTIFF (LRO NAC / Base)", type=["tif", "tiff", "geotiff", "png", "jpg"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Alignment Engine Parameters")
conf_thresh = st.sidebar.slider("LoFTR Confidence Threshold (τ)", 0.05, 0.90, 0.15, step=0.05)
anms_cap = st.sidebar.slider("ANMS Cap per Cell (8x8 Grid)", 1, 12, 4, step=1)
enable_subpixel = st.sidebar.checkbox("2D Parabolic Taylor Sub-Pixel Peak Refinement", value=True)
magsac_thresh = st.sidebar.slider("USAC-MAGSAC++ Reprojection Threshold (px)", 0.5, 3.0, 1.5, step=0.25)

# -----------------------------------------------------------------------------
# Ingestion & Data Preparation
# -----------------------------------------------------------------------------

img_source: Optional[np.ndarray] = None
img_ref: Optional[np.ndarray] = None

if selected_key in ["scenario_a", "scenario_b", "scenario_c"]:
    bm = benchmarks.get(selected_key, {})
    sample_dir = get_sample_data_dir()
    src_path = sample_dir / bm["source"]["filename"]
    ref_path = sample_dir / bm["reference"]["filename"]

    try:
        img_source = load_geotiff_file(src_path)
        img_ref = load_geotiff_file(ref_path)
    except Exception as e:
        st.error(f"Failed to load cached benchmark GeoTIFF: {e}")

    # Display Mission Acquisition Metadata Card
    st.markdown(f"### 🎯 Benchmark: {bm.get('title', selected_benchmark)}")
    st.markdown(f"**Target Site**: {bm.get('target', 'Moon')} — *{bm.get('description', '')}*")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.info(f"""
        **🚀 Source: {bm['source']['spacecraft']}**
        - **Payload Sensor**: `{bm['source']['sensor']}`
        - **Ground Sampling Distance**: `{bm['source']['gsd_m']} m/pixel`
        - **Orbit Altitude**: `{bm['source']['altitude_km']} km`
        - **Solar Geometry**: Azimuth `{bm['source']['sun_azimuth_deg']}°`, Elevation `{bm['source']['sun_elevation_deg']}°`
        """)
    with col_meta2:
        st.info(f"""
        **🔭 Reference: {bm['reference']['spacecraft']}**
        - **Payload Sensor**: `{bm['reference']['sensor']}`
        - **Ground Sampling Distance**: `{bm['reference']['gsd_m']} m/pixel`
        - **Orbit Altitude**: `{bm['reference']['altitude_km']} km`
        - **Solar Geometry**: Azimuth `{bm['reference']['sun_azimuth_deg']}°`, Elevation `{bm['reference']['sun_elevation_deg']}°`
        """)

elif selected_key == "synthetic_sim":
    sim = LunarTerrainSimulator(size=(256, 256), seed=101)
    dem = sim.generate_dem(num_craters=12)

    sun_ref = SunAngles(azimuth_deg=60.0, elevation_deg=25.0)
    img_ref = sim.render_optical_image(dem, sun_ref)

    sun_src = SunAngles(azimuth_deg=240.0, elevation_deg=25.0)
    img_src_raw = sim.render_optical_image(dem, sun_src)

    mat_true = cv2.getRotationMatrix2D((128, 128), 1.8, 1.0)
    mat_true[:, 2] += [4.5, -3.2]
    img_source = cv2.warpAffine(img_src_raw, mat_true, (256, 256))
    st.info(r"💡 Running in **Synthetic Mode**: Simulating Chandrayaan-2 OHRC morning frame vs. LRO NAC afternoon frame with $180^\circ$ inverted shadow polarity.")

elif selected_key == "custom_upload":
    if uploaded_src is not None and uploaded_ref is not None:
        try:
            img_source = load_uploaded_image(uploaded_src)
            img_ref = load_uploaded_image(uploaded_ref)
            st.success(f"Loaded Custom Source ({img_source.shape[1]}x{img_source.shape[0]}) and Reference ({img_ref.shape[1]}x{img_ref.shape[0]}) GeoTIFFs.")
        except Exception as e:
            st.error(f"Error reading GeoTIFF rasters: {e}")
    else:
        st.warning("Please upload both Source and Reference GeoTIFFs via the sidebar to execute registration.")

# -----------------------------------------------------------------------------
# End-to-End Alignment Pipeline Execution with Progress Bar
# -----------------------------------------------------------------------------

if img_source is not None and img_ref is not None:
    col_btn, _ = st.columns([1, 3])
    run_alignment = col_btn.button("🚀 Execute Multi-Modal Alignment Pipeline", type="primary", use_container_width=True)

    if run_alignment:
        progress_bar = st.progress(0, text="Initializing planetary registration engine...")

        # Step 1: Preprocessing & GeoTIFF normalizations
        progress_bar.progress(15, text="Step 1/5: Ingesting & normalizing GeoTIFF dynamic ranges...")
        time.sleep(0.1)

        # Step 2: Illumination-Invariant Log-Gabor Phase Congruency
        progress_bar.progress(35, text="Step 2/5: Vectorized 2D Log-Gabor Phase Congruency (PyTorch FFT)...")
        pc_engine = PhaseCongruencyEngine(num_scales=4, num_orientations=6)
        pc_src = pc_engine.compute(img_source)
        pc_ref = pc_engine.compute(img_ref)
        time.sleep(0.1)

        # Step 3: Dense LoFTR Cross-Attention Matching
        progress_bar.progress(60, text="Step 3/5: Dense Keypoint Extraction via kornia.feature.LoFTR...")
        start_t = time.perf_counter()
        matcher = DenseLoFTRMatcher(
            pretrained="outdoor",
            confidence_threshold=conf_thresh,
            grid_bins=8,
            cap_per_cell=anms_cap,
            magsac_reproj_threshold=magsac_thresh,
        )

        src_tensor, norm_src = matcher.prepare_geotiff_array(pc_src.max_moment)
        ref_tensor, norm_ref = matcher.prepare_geotiff_array(pc_ref.max_moment)

        raw_matches = matcher.extract_dense_correspondences(
            src_tensor, ref_tensor, norm_src.shape, norm_ref.shape
        )

        # Step 4: 8x8 Grid ANMS & 2D Parabolic Taylor Sub-pixel Refinement
        progress_bar.progress(80, text="Step 4/5: Grid-Based ANMS (8x8 Grid) & 2D Parabolic Taylor Refinement...")
        anms_matches = matcher.apply_grid_anms_8x8(raw_matches, norm_ref.shape)

        if enable_subpixel and anms_matches:
            refined_matches = matcher.refine_subpixel_taylor_2d(anms_matches, norm_src, norm_ref)
        else:
            refined_matches = anms_matches

        # Step 5: USAC-MAGSAC Homography & Warping
        progress_bar.progress(95, text="Step 5/5: USAC-MAGSAC++ Homography Estimation & Warping...")
        inliers, H, warped_source = matcher.filter_outliers_magsac(
            refined_matches, img_source, img_ref.shape
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        # Step 6: Evaluation Diagnostics
        report = EvaluationEngine.generate_report(
            total_matches=len(raw_matches),
            inliers=inliers,
            image_shape=img_ref.shape,
            homography=H,
            processing_time_ms=elapsed_ms,
        )

        progress_bar.progress(100, text="Alignment Complete! Generated Hackathon Diagnostics.")
        time.sleep(0.2)
        progress_bar.empty()

        # Cache in Streamlit session state
        st.session_state["result_report"] = report
        st.session_state["inliers"] = inliers
        st.session_state["raw_matches"] = raw_matches
        st.session_state["homography"] = H
        st.session_state["warped_source"] = warped_source
        st.session_state["img_source"] = img_source
        st.session_state["img_ref"] = img_ref

# -----------------------------------------------------------------------------
# Results Presentation: Scorecards, Plots & Blending
# -----------------------------------------------------------------------------

if "result_report" in st.session_state:
    report: RegistrationEvaluationReport = st.session_state["result_report"]
    inliers = st.session_state["inliers"]
    H = st.session_state["homography"]
    warped_src = st.session_state["warped_source"]
    img_source = st.session_state["img_source"]
    img_ref = st.session_state["img_ref"]

    st.markdown("---")
    st.subheader("📊 Planetary Hackathon KPI Metric Scorecards")

    # 5 KPI Metric Scorecards
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric(
        "Sub-Pixel RMSE",
        f"{report.rmse_pixels:.3f} px",
        delta=f"{report.rmse_pixels - 0.40:.3f} vs 0.40 px target",
        delta_color="inverse",
    )
    col_k2.metric("Inlier Count", f"{report.inlier_count} pts")
    col_k3.metric("Inlier Ratio", f"{report.inlier_ratio_percent:.1f}%")
    col_k4.metric(
        "Spatial Entropy",
        f"{report.spatial_uniformity_entropy:.3f} / 1.0",
        help="2D Shannon Spatial Entropy across 8x8 grid (1.0 = optimal non-clumping across crater rims)",
    )
    col_k5.metric("Pipeline Latency", f"{report.processing_time_ms:.1f} ms")

    if report.meets_isro_mandate:
        st.success("🎯 **ISRO SIH PS 26166 Mandate Passed**: Sub-pixel registration RMSE < 0.40 pixels achieved under extreme opposite lighting!")
    else:
        st.warning("⚠️ Sub-pixel RMSE threshold (> 0.40 px) or inlier count requires refinement.")

    # 4 Interactive Inspection Tabs
    tab_overlap, tab_tiepoints, tab_diagnostics, tab_exports = st.tabs([
        "🔬 Crater Rim Overlap (Wipe & Checkerboard)",
        "🔗 Side-by-Side Tie-Point Correspondence Plot",
        "📈 Residual Error Scatter & Frequency Invariance",
        "📥 Structured JSON & GIS Tie-Point Export",
    ])

    # TAB 1: 50/50 Checkerboard Blend and Sliding Wipe Tool
    with tab_overlap:
        st.markdown("### 🎚️ Interactive Sliding Wipe Tool (Crater Rim Overlap Inspection)")
        st.caption("Drag the wipe slider from 0% to 100% across the boundary. Continuous crater boundaries confirm sub-pixel planetary alignment.")

        if warped_src is not None:
            split_pct = st.slider("Sliding Wipe Divider Position (%)", 0, 100, 50, step=1)
            h, w = img_ref.shape
            col_split = int((split_pct / 100.0) * w)

            swipe_composite = np.zeros_like(img_ref)
            swipe_composite[:, :col_split] = img_ref[:, :col_split]
            swipe_composite[:, col_split:] = warped_src[:, col_split:]

            # Draw divider line
            swipe_rgb = cv2.cvtColor((swipe_composite * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            if 0 <= col_split < w:
                swipe_rgb[:, col_split, :] = [255, 40, 40]  # Red divider

            st.image(
                swipe_rgb,
                caption=f"Left: Reference Frame | Right: Registered Warped Source (Divider at {split_pct}%)",
                width="stretch",
            )

        st.markdown("---")
        st.markdown("### 🏁 50/50 Checkerboard Blend")
        st.caption("Alternating spatial tiles verify seamless crater rim boundaries and wall continuity across frames.")

        if warped_src is not None:
            tile_size = st.slider("Checkerboard Tile Size (pixels)", 16, 64, 32, step=8)
            h, w = img_ref.shape
            checker = np.zeros_like(img_ref)
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                        checker[y : y + tile_size, x : x + tile_size] = img_ref[y : y + tile_size, x : x + tile_size]
                    else:
                        checker[y : y + tile_size, x : x + tile_size] = warped_src[y : y + tile_size, x : x + tile_size]

            st.image(checker, caption=f"50/50 Checkerboard Blend ({tile_size}x{tile_size} px tiles)", width="stretch", clamp=True)

    # TAB 2: Interactive Side-by-Side Tie-Point Correspondence Plot
    with tab_tiepoints:
        st.subheader("🔗 Interactive Tie-Point Correspondence Plot")
        st.caption("Side-by-side view connecting Source (Left) to Reference (Right), color-coded by LoFTR match confidence.")

        max_pts = st.slider("Max Tie-Points to Display", 10, 150, 40, step=5)
        fig_corr = render_tie_point_correspondences(img_source, img_ref, inliers, max_display=max_pts)
        st.pyplot(fig_corr)

    # TAB 3: Residual Error Scatter & Frequency Invariance
    with tab_diagnostics:
        st.subheader("📈 Residual Error Scatter Field & Frequency Invariance")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("#### Reprojection Residual Scatter Field")
            plot_tmp_path = Path("tests/temp_scatter_display.png")
            report.export_residual_scatter_plot(plot_tmp_path, background_image=img_ref)
            st.image(str(plot_tmp_path), width="stretch")

        with col_d2:
            st.markdown("#### Log-Gabor Phase Congruency (M_max)")
            pc_e = PhaseCongruencyEngine(num_scales=4, num_orientations=6)
            pc_out = pc_e.compute(img_source)
            st.image(pc_out.max_moment, caption="Maximum Moment (M_max) Invariant Step Edges", width="stretch", clamp=True)

    # TAB 4: Structured JSON & GIS Export
    with tab_exports:
        st.subheader("📥 Structured JSON Report & Photogrammetry Export")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("#### Structured JSON Evaluation Report")
            json_str = json.dumps(report.to_dict(), indent=2)
            st.download_button(
                label="📄 Download JSON Evaluation Report",
                data=json_str,
                file_name="lunar_core_evaluation_report.json",
                mime="application/json",
            )
            st.code(json_str[:600] + "\n  ...\n}", language="json")

        with col_ex2:
            st.markdown("#### Ground Control Points (GCP CSV)")
            gcp_csv_lines = ["gcp_id,ref_x,ref_y,src_x,src_y,reproj_ref_x,reproj_ref_y,residual_px,confidence\n"]
            for pt in report.tie_points:
                gcp_csv_lines.append(
                    f"{pt['id']},{pt['ref_x']:.4f},{pt['ref_y']:.4f},{pt['src_x']:.4f},{pt['src_y']:.4f},"
                    f"{pt['reprojected_ref_x']:.4f},{pt['reprojected_ref_y']:.4f},{pt['residual_pixels']:.4f},{pt['confidence']:.4f}\n"
                )
            gcp_csv = "".join(gcp_csv_lines)
            st.download_button(
                label="🗺️ Download GCP CSV (QGIS / ArcGIS / ISIS3)",
                data=gcp_csv,
                file_name="planetary_gcps.csv",
                mime="text/csv",
            )
            st.code("".join(gcp_csv_lines[:6]), language="csv")
