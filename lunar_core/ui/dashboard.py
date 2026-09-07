"""
Interactive Lunar Registration Dashboard (Streamlit UI).
Adheres to Clean Architecture and SIH PS 26166 Hackathon Evaluation Metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path regardless of execution working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from lunar_core.models import SensorModality, SunAngles, TransformationType
from lunar_core.pipeline import LunarCorePipeline
from lunar_core.data_io.raster_writer import PlanetaryRasterWriter
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator


st.set_page_config(
    page_title="ISRO Chandrayaan-2 Lunar Registration Engine",
    page_icon="🌙",
    layout="wide",
)

st.title("🌙 ISRO Chandrayaan-2 Lunar Image Registration Engine")
st.markdown("### SIH PS 26166: Multi-Modal, Sun-Angle and Scale-Invariant Image Correspondence")

# Sidebar Configuration
st.sidebar.header("🛰️ Planetary Mission Configuration")
modality_ref = st.sidebar.selectbox("Reference Modality", [m.value for m in SensorModality], index=0)
modality_tgt = st.sidebar.selectbox("Target Modality", [m.value for m in SensorModality], index=1)

st.sidebar.markdown("---")
st.sidebar.header("☀️ Solar Geometry Controls")
ref_az = st.sidebar.slider("Reference Sun Azimuth (°)", 0.0, 360.0, 60.0, step=5.0)
ref_el = st.sidebar.slider("Reference Sun Elevation (°)", 5.0, 85.0, 25.0, step=5.0)
tgt_az = st.sidebar.slider("Target Sun Azimuth (°)", 0.0, 360.0, 240.0, step=5.0)
tgt_el = st.sidebar.slider("Target Sun Elevation (°)", 5.0, 85.0, 35.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Transformation & Matching")
trans_type_str = st.sidebar.selectbox("Geometric Model", ["homography", "affine"], index=0)
trans_type = TransformationType(trans_type_str)
enable_subpixel = st.sidebar.checkbox("Enable 2D Quadratic Sub-Pixel Refinement", value=True)
enable_anms = st.sidebar.checkbox("Enable Grid-Based ANMS Uniform Distributor", value=True)

# Generate Synthetic Pair with Ground-Truth
sim = LunarTerrainSimulator(size=(320, 320), seed=42)
sun_ref = SunAngles(azimuth_deg=ref_az, elevation_deg=ref_el)
sun_tgt = SunAngles(azimuth_deg=tgt_az, elevation_deg=tgt_el)

with st.spinner("Rendering lunar crater terrain & executing clean registration pipeline..."):
    img_ref, img_tgt, true_affine, _ = sim.generate_registered_pair_with_ground_truth(
        sun_ref=sun_ref,
        sun_tgt=sun_tgt,
        true_translation=(6.0, -4.0),
        true_rotation_deg=2.5,
    )

    pipeline = LunarCorePipeline(
        transformation_type=trans_type,
        enable_photometric=True,
        enable_anms=enable_anms,
        enable_subpixel=enable_subpixel,
    )
    result = pipeline.register(
        ref_image=img_ref,
        target_image=img_tgt,
        ref_sun=sun_ref,
        target_sun=sun_tgt,
    )

m = result.metrics

# Hackathon Real-Time Benchmark Cards
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
col_m1.metric("RMSE (Sub-pixel)", f"{m.rmse_pixels:.3f} px", delta=f"{m.rmse_pixels - 0.4:.3f} vs 0.4 px", delta_color="inverse")
col_m2.metric("Total Matches", f"{m.total_matches}")
col_m3.metric("Inlier Count", f"{m.inlier_count}")
col_m4.metric("Inlier Ratio (%)", f"{m.inlier_ratio * 100:.1f}%")
col_m5.metric("Spatial Uniformity", f"{m.spatial_uniformity_entropy:.2f} / 1.0")
col_m6.metric("Pipeline Latency", f"{m.processing_time_ms:.1f} ms")

if m.meets_isro_mandate:
    st.success("🎯 **ISRO SIH PS 26166 Mandate Passed**: Sub-pixel RMSE < 0.40 pixels achieved under extreme illumination reversal.")
else:
    st.warning("⚠️ High geometric residual detected or insufficient inliers.")

# Main Display Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Registration & Blending",
    "🔗 Interactive Vector Flow Field",
    "🌊 Phase Congruency & Retinex",
    "🗺️ GIS & Photogrammetry Export",
])

with tab1:
    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.subheader("Reference Image (Morning Sun)")
    col_i1.image(img_ref, clamp=True, use_container_width=True, caption=f"Sun Azimuth: {ref_az}°, Elevation: {ref_el}°")

    col_i2.subheader("Target Image (Afternoon Sun)")
    col_i2.image(img_tgt, clamp=True, use_container_width=True, caption=f"Sun Azimuth: {tgt_az}°, Elevation: {tgt_el}° (Shadow Inverted)")

    col_i3.subheader("Registered & Warped Result")
    if result.warped_target is not None:
        col_i3.image(result.warped_target, clamp=True, use_container_width=True, caption="Aligned to Reference Frame")
    else:
        col_i3.error("Transformation could not be computed.")

    st.markdown("### 🎚️ 50/50 Split-Swipe Comparison")
    st.caption("Slide to wipe across Reference (Left) and Aligned Target (Right) to visually confirm sub-pixel alignment along crater boundaries.")
    if result.warped_target is not None:
        split_pct = st.slider("Split-Swipe Divider Position (%)", 0, 100, 50)
        h, w = img_ref.shape
        split_col = int((split_pct / 100.0) * w)
        split_view = np.zeros_like(img_ref)
        split_view[:, :split_col] = img_ref[:, :split_col]
        split_view[:, split_col:] = result.warped_target[:, split_col:]
        split_rgb = cv2.cvtColor((split_view * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if 0 <= split_col < w:
            split_rgb[:, split_col, :] = [255, 30, 30]
        st.image(split_rgb, use_container_width=True, caption=f"Left: Reference Frame | Right: Registered Target (Split at {split_pct}%)")

    st.markdown("### 🏁 Checkerboard Blending Inspection")
    st.caption("Alternating spatial tiles verify seamless crater rim boundaries and wall continuity.")
    if result.warped_target is not None:
        checker_size = st.slider("Checkerboard Tile Size (pixels)", 16, 64, 32, step=8)
        h, w = img_ref.shape
        checker = np.zeros_like(img_ref)
        for y in range(0, h, checker_size):
            for x in range(0, w, checker_size):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    checker[y:y+checker_size, x:x+checker_size] = img_ref[y:y+checker_size, x:x+checker_size]
                else:
                    checker[y:y+checker_size, x:x+checker_size] = result.warped_target[y:y+checker_size, x:x+checker_size]
        st.image(checker, clamp=True, use_container_width=True)

with tab2:
    st.subheader("Interactive Vector Flow Field (Match Directions & Magnitudes)")
    st.caption("Visualizes displacement vectors $(\Delta x, \Delta y)$ with color-coded magnitude and angular orientation.")

    if result.inliers:
        rx_list = np.array([m.ref_xy[0] for m in result.inliers])
        ry_list = np.array([m.ref_xy[1] for m in result.inliers])
        dx_list = np.array([m.target_xy[0] - m.ref_xy[0] for m in result.inliers])
        dy_list = np.array([m.target_xy[1] - m.ref_xy[1] for m in result.inliers])
        magnitudes = np.sqrt(dx_list**2 + dy_list**2)
        angles_deg = np.degrees(np.arctan2(dy_list, dx_list))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_ref, cmap="gray")

        q = ax.quiver(
            rx_list, ry_list, dx_list, dy_list,
            magnitudes,
            cmap="autumn",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            headwidth=4,
            headlength=5,
        )
        cbar = plt.colorbar(q, ax=ax, shrink=0.7)
        cbar.set_label("Displacement Magnitude (pixels)")
        ax.scatter(rx_list, ry_list, c="lime", s=25, edgecolors="black", linewidths=0.5, label="Inlier Points")
        ax.set_title(f"USAC-MAGSAC++ Vector Flow Field ({len(result.inliers)} Inliers)")
        ax.legend(loc="upper right")
        ax.axis("off")
        st.pyplot(fig)

        c_v1, c_v2, c_v3 = st.columns(3)
        c_v1.metric("Mean Displacement", f"{np.mean(magnitudes):.2f} px")
        c_v2.metric("Max Displacement", f"{np.max(magnitudes):.2f} px")
        c_v3.metric("Mean Flow Direction", f"{np.mean(angles_deg):.1f}°")
    else:
        st.warning("No inliers detected for vector flow field.")

with tab3:
    st.subheader("Illumination-Invariant Representations")
    pc_out = pipeline.pc_engine.compute(img_ref)
    msr_img = pipeline.contrast.process(img_ref)

    c_pc1, c_pc2, c_pc3 = st.columns(3)
    c_pc1.image(pc_out.phase_congruency, clamp=True, use_container_width=True, caption="Total Phase Congruency (PC)")
    c_pc2.image(pc_out.max_moment, clamp=True, use_container_width=True, caption="Maximum Moment (M_max - Step Edges)")
    c_pc3.image(msr_img, clamp=True, use_container_width=True, caption="Multi-Scale Retinex (MSR) Range Compression")

with tab4:
    st.subheader("Planetary GIS & Photogrammetry Export")
    st.markdown("Export ground-truth tie-points and displacement vector fields for **QGIS**, **ArcGIS Pro**, or **USGS ISIS3 Jigsaw**.")

    gcp_lines = ["gcp_id,pixel_ref,line_ref,pixel_tgt,line_tgt,residual_px,confidence\n"]
    for idx, m_pt in enumerate(result.inliers):
        res_val = m_pt.residual_error if m_pt.residual_error is not None else 0.0
        gcp_lines.append(f"{idx},{m_pt.ref_xy[0]:.4f},{m_pt.ref_xy[1]:.4f},{m_pt.target_xy[0]:.4f},{m_pt.target_xy[1]:.4f},{res_val:.4f},{m_pt.confidence:.4f}\n")
    gcp_csv = "".join(gcp_lines)

    import json
    features = []
    for idx, m_pt in enumerate(result.inliers):
        features.append({
            "type": "Feature",
            "id": idx,
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(m_pt.ref_xy[0]), float(m_pt.ref_xy[1])], [float(m_pt.target_xy[0]), float(m_pt.target_xy[1])]],
            },
            "properties": {
                "gcp_id": idx,
                "dx": float(m_pt.target_xy[0] - m_pt.ref_xy[0]),
                "dy": float(m_pt.target_xy[1] - m_pt.ref_xy[1]),
                "residual_px": float(m_pt.residual_error if m_pt.residual_error is not None else 0.0),
            },
        })
    geojson_str = json.dumps({"type": "FeatureCollection", "features": features}, indent=2)

    col_e1, col_e2 = st.columns(2)
    col_e1.download_button(
        label="📥 Download Ground Control Points (GCPs CSV)",
        data=gcp_csv,
        file_name="lunar_core_gcps.csv",
        mime="text/csv",
    )
    col_e2.download_button(
        label="📥 Download Displacement Vectors (GeoJSON)",
        data=geojson_str,
        file_name="lunar_core_vector_field.geojson",
        mime="application/geo+json",
    )
