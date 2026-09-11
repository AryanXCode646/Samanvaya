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
import tempfile
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
from lunar_core.alignment.rift_matcher import ClassicalRIFTMatcher
from lunar_core.preprocessing.phase_congruency import PhaseCongruencyEngine
from lunar_core.preprocessing.photometric import PhotometricNormalizer
from lunar_core.evaluation.metrics import EvaluationEngine, RegistrationEvaluationReport
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator
from lunar_core.data_io.raster_reader import PlanetaryRasterReader
from lunar_core.data_io.archive_status import (
    OFFICIAL_ARCHIVES,
    MissionArchiveStatus,
    check_archive,
    latest_product,
    scan_local_chandrayaan2,
    status_with_local_products,
)


# Configure Streamlit Page
st.set_page_config(
    page_title="ISRO Chandrayaan-2 Lunar Alignment Portal",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg-base: #07141d;
        --bg-panel: rgba(11, 20, 30, 0.88);
        --bg-panel-strong: rgba(14, 26, 38, 0.98);
        --border-soft: rgba(147, 186, 255, 0.18);
        --border-strong: rgba(147, 186, 255, 0.33);
        --text-primary: #ecf7ff;
        --text-muted: #b3cfee;
        --text-faint: #8aa7c8;
        --accent: #73c8ff;
        --accent-strong: #8ddcff;
        --success: #6fe3a4;
        --warning: #f9d871;
        --danger: #ff7a7a;
        --shadow: rgba(0, 0, 0, 0.35);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(116, 182, 255, 0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(99, 236, 197, 0.08), transparent 26%),
            linear-gradient(180deg, #06121a 0%, #0a1723 40%, #091720 100%);
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 17, 25, 0.99), rgba(10, 20, 28, 0.98));
        border-right: 1px solid var(--border-soft);
        box-shadow: 8px 0 24px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stSidebar"] .block-container {
        background: transparent;
        padding-top: 1.2rem;
    }

    .portal-shell {
        background: linear-gradient(180deg, rgba(12, 21, 32, 0.92), rgba(13, 23, 33, 0.86));
        border: 1px solid var(--border-strong);
        border-radius: 22px;
        padding: 1.25rem 1.45rem 1rem;
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        margin-bottom: 1.25rem;
    }

    .portal-shell h1 {
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.04em;
        color: var(--text-primary);
        text-shadow: 0 0 18px rgba(116, 200, 255, 0.18);
    }

    .portal-shell p {
        margin-top: 0.55rem;
        color: var(--text-muted);
        font-size: 1.02rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(18, 35, 49, 0.96), rgba(10, 21, 30, 0.96));
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 12px 30px rgba(5, 12, 18, 0.25), inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: linear-gradient(180deg, rgba(98, 223, 164, 0.18), rgba(98, 223, 164, 0.12));
        border: 1px solid rgba(111, 227, 164, 0.65);
        color: #d9ffe9;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15, 29, 39, 0.96), rgba(10, 20, 28, 0.96));
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.18);
        padding: 0.8rem 0.9rem;
    }

    div[data-testid="stMetric"] label {
        color: var(--text-faint) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        margin-top: 0.15rem;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: var(--text-muted) !important;
        font-size: 0.77rem !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(134, 191, 255, 0.38) !important;
        background: linear-gradient(180deg, rgba(20, 42, 62, 0.98), rgba(12, 25, 38, 0.98)) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 22px rgba(0,0,0,0.18);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .stButton > button:hover {
        border-color: rgba(141, 220, 255, 0.8) !important;
        box-shadow: 0 14px 28px rgba(26, 96, 161, 0.22) !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, rgba(75, 170, 255, 0.25), rgba(27, 71, 128, 0.9)) !important;
        border-color: rgba(119, 199, 255, 0.8) !important;
    }

    .stTabs [role="tablist"] {
        border-bottom: 1px solid var(--border-soft);
        gap: 0.55rem;
    }

    .stTabs [role="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 0.7rem 1rem;
        color: var(--text-muted) !important;
        font-weight: 600;
        background: rgba(255,255,255,0.02);
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, rgba(26, 42, 57, 0.95), rgba(16, 28, 39, 0.92));
        border: 1px solid var(--border-strong);
        border-bottom: none;
        color: var(--text-primary) !important;
    }

    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 176, 230, 0.18) !important;
        box-shadow: 0 8px 18px rgba(0,0,0,0.14);
    }

    textarea, input, select {
        border-radius: 10px !important;
        border: 1px solid rgba(148, 176, 230, 0.22) !important;
        background: rgba(10, 20, 28, 0.95) !important;
        color: var(--text-primary) !important;
    }

    .stSlider > div > div {
        background: rgba(255,255,255,0.02);
    }

    .stDownloadButton > button {
        background: linear-gradient(180deg, rgba(65, 90, 116, 0.98), rgba(26, 39, 54, 0.98)) !important;
        border: 1px solid rgba(144, 182, 240, 0.32) !important;
        color: var(--text-primary) !important;
    }

    .stFileUploader > section {
        background: rgba(10, 19, 26, 0.9);
        border: 1px dashed rgba(141, 197, 255, 0.3);
        border-radius: 14px;
    }

    .sidebar .stSelectbox, .sidebar .stSlider {
        padding-bottom: 0.3rem;
    }

    .main .stMarkdown {
        color: var(--text-primary);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selected_key = "scenario_a"
img_source = None
img_ref = None
summary_badge = "SYSTEM READY"

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


def parse_uploaded_pds4_metadata(uploaded_file) -> Tuple[SunAngles, float, SensorModality]:
    """Parse an uploaded PDS4 XML label through the hardened raster reader."""
    suffix = Path(uploaded_file.name or "metadata.xml").suffix.lower() or ".xml"
    with tempfile.NamedTemporaryFile(prefix="samanvaya-pds4-", suffix=suffix, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(uploaded_file.getvalue())
    try:
        return PlanetaryRasterReader.parse_pds4_metadata(temp_path, allowed_dir=temp_path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def moon_preview_image(image: np.ndarray) -> np.ndarray:
    """Return a realistic lunar-style visualization instead of raw grayscale microworking."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        vmin = float(np.percentile(valid, 1.0))
        vmax = float(np.percentile(valid, 99.5))
        denom = max(vmax - vmin, 1e-6)
        norm = np.clip((arr - vmin) / denom, 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        rgb[..., 0] = np.clip(rgb[..., 0] * 0.78 + 18.0, 0.0, 100.0)
        rgb[..., 1] = np.clip(rgb[..., 1] * 1.05 + 8.0, -127.0, 128.0)
        rgb[..., 2] = np.clip(rgb[..., 2] * 1.15 - 12.0, -127.0, 128.0)
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return rgb
    if arr.ndim == 3 and arr.shape[2] == 1:
        return moon_preview_image(arr[:, :, 0])
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = np.clip(arr, 0.0, 1.0)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255.0).astype(np.uint8)
        return rgb[..., :3]
    return np.zeros((1, 1, 3), dtype=np.uint8)


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


def data_type_label_for_key(selected_key: str) -> str:
    """Return the current dataset class in a UI-safe, non-misleading label."""
    if selected_key == "custom_upload":
        return "REAL MISSION DATA"
    if selected_key == "synthetic_sim":
        return "DEMO / SYNTHETIC"
    return "DEMO / SYNTHETIC"


def format_unknown(value, default: str = "Unknown") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value if value.strip() else default
    return str(value)


def preview_for_display(image: Optional[np.ndarray], max_side: int = 1200) -> Optional[np.ndarray]:
    """Downsample display imagery without changing the scientific input array."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2:
        return None
    height, width = arr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return arr
    return cv2.resize(arr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def render_image_panel(
    title: str,
    state_label: str,
    image: Optional[np.ndarray],
    caption: str,
    data_type: str,
) -> None:
    """Render a truthful, aspect-preserving image panel for the main workspace."""
    st.markdown(f"#### {title}")
    st.caption(f"{state_label} · {data_type}")
    if image is None:
        st.info(caption)
        return
    preview = preview_for_display(image)
    st.image(preview, caption=caption, width="stretch")
    st.caption(f"Preview resolution: {preview.shape[1]} × {preview.shape[0]} px")


def clear_registration_state() -> None:
    """Discard outputs that belong to a previous image pair or configuration."""
    for key in (
        "result_report",
        "inliers",
        "raw_matches",
        "homography",
        "matcher_path",
        "warped_source",
        "img_source",
        "img_ref",
        "backend_trace",
        "last_run_summary",
    ):
        st.session_state.pop(key, None)


def load_local_demo_evidence() -> dict:
    """Load the locally verified benchmark summary if it exists."""
    evidence_path = _PROJECT_ROOT / "output" / "demo_run" / "samanvaya_evaluation_report.json"
    if not evidence_path.exists():
        return {}
    try:
        with evidence_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def scan_local_products_cached(root: str) -> list[dict]:
    """Cache metadata-only local discovery, never raster arrays."""
    path = Path(root)
    if not path.exists():
        return []
    return [product.to_dict() for product in scan_local_chandrayaan2(path)]


def product_from_dict(record: dict) -> object:
    """Rehydrate a catalog record for UI selection without broad model coupling."""
    from lunar_core.data_io.mission_product import MissionProduct

    values = dict(record)
    for key in ("image_path", "label_path"):
        if values.get(key):
            values[key] = Path(values[key])
    return MissionProduct(**{key: value for key, value in values.items() if key in MissionProduct.__dataclass_fields__})


def render_workflow_steps(active_index: int = 1) -> None:
    steps = ["01 DATA", "02 PAIR", "03 REGISTER", "04 VERIFY"]
    cols = st.columns(len(steps))
    for idx, label in enumerate(steps):
        with cols[idx]:
            style = "background: rgba(115,200,255,0.14); border:1px solid rgba(115,200,255,0.35); color:#dff6ff;"
            if idx == active_index:
                style = "background: rgba(111,227,164,0.12); border:1px solid rgba(111,227,164,0.4); color:#eafff1;"
            elif idx < active_index:
                style = "background: rgba(255,255,255,0.03); border:1px solid rgba(148,176,230,0.20); color:#d9ebff;"
            st.markdown(
                f"<div style='padding:0.65rem 0.7rem; border-radius:12px; text-align:center; font-size:0.79rem; font-weight:700; letter-spacing:0.08em; {style}'>{label}</div>",
                unsafe_allow_html=True,
            )


def render_pair_card(title: str, mission: str, instrument: str, product_id: str, dimensions: str, gsd: str, status: str) -> None:
    st.markdown(
        f"""
        <div style="padding:0.9rem 1rem; border:1px solid rgba(148,176,230,0.24); border-radius:16px; background:linear-gradient(180deg, rgba(12,23,33,0.95), rgba(9,18,27,0.94)); min-height: 188px;">
            <div style="font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase; color:#8fb9d8; margin-bottom:0.7rem;">{title}</div>
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.18rem;">{format_unknown(mission)}</div>
            <div style="font-size:0.95rem; color:#d5ebff; margin-bottom:0.7rem;">{format_unknown(instrument)}</div>
            <div style="margin-bottom:0.38rem; color:#edfaff; font-weight:600;">{format_unknown(product_id)}</div>
            <div style="margin-bottom:0.38rem; color:#dcecff;">{format_unknown(dimensions)}</div>
            <div style="margin-bottom:0.38rem; color:#dcecff;">{format_unknown(gsd)}</div>
            <div style="margin-top:0.7rem; display:inline-flex; padding:0.3rem 0.55rem; border-radius:999px; background: rgba(255,255,255,0.03); border:1px solid rgba(148,176,230,0.2); font-size:0.72rem; color:#eafff1;">{format_unknown(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_launch_modal(selected_benchmark: str, selected_key: str, source_modality: str, ref_modality: str) -> None:
    """Render a modal-style confirmation dialog before launch."""
    if not hasattr(st, "dialog"):
        if not st.session_state.get("show_launch_modal", False):
            return
        st.markdown(
            """
            <div style='position: fixed; inset: 0; background: rgba(5,10,15,0.72); z-index: 999; display: flex; align-items: center; justify-content: center;'>
                <div style='width: min(620px, 92vw); background: rgba(14,23,33,0.98); border: 1px solid rgba(160,204,255,0.25); border-radius: 18px; padding: 1.5rem; box-shadow: 0 24px 80px rgba(0,0,0,0.45);'>
                    <h3 style='margin:0 0 0.5rem; color:#eef8ff;'>Launch alignment</h3>
                    <p style='margin:0 0 0.75rem; color:#c6ddf9;'>Selected benchmark: <strong>""" + selected_benchmark + """</strong></p>
                    <div style='display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom: 1rem;'>
                        <span style='padding:0.35rem 0.7rem; border-radius:999px; background: rgba(97,186,255,0.1); border:1px solid rgba(97,186,255,0.35); color:#d9f3ff;'>""" + source_modality + """ → """ + ref_modality + """</span>
                        <span style='padding:0.35rem 0.7rem; border-radius:999px; background: rgba(76,201,140,0.1); border:1px solid rgba(76,201,140,0.35); color:#d7ffea;'>""" + selected_key + """</span>
                    </div>
                    <p style='margin:0; color:#dfeefc;'>This will run the full multi-modal alignment pipeline, including phase congruency, LoFTR matching, ANMS, and validation diagnostics.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Confirm & Run", type="primary", use_container_width=True):
                st.session_state["show_launch_modal"] = False
                st.session_state["launch_confirmed"] = True
                st.rerun()
        with col_cancel:
            if st.button("Cancel", type="secondary", use_container_width=True):
                st.session_state["show_launch_modal"] = False
                st.session_state["launch_confirmed"] = False
                st.rerun()
        st.stop()
        return

    @st.dialog("Launch alignment pipeline")
    def _modal():
        st.markdown(f"**Selected benchmark:** {selected_benchmark}")
        st.write(f"Source: {source_modality} → Reference: {ref_modality}")
        st.write(f"Configuration: {selected_key}")
        st.caption("This will run the full multi-modal alignment workflow, including solar normalization, phase congruency, dense matching, and scientific validation reporting.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Confirm & Run", type="primary", use_container_width=True):
                st.session_state["launch_confirmed"] = True
                st.rerun()
        with col_b:
            if st.button("Cancel", type="secondary", use_container_width=True):
                st.session_state["launch_confirmed"] = False
                st.rerun()

    if st.session_state.get("show_launch_modal", False):
        _modal()


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

local_demo = load_local_demo_evidence()

st.sidebar.header("MODE")
st.sidebar.caption("Scientifically honest workflow selection")
mode_choice = st.sidebar.radio("Data source", ["Real Mission Data", "Synthetic / Benchmark"], index=0 if selected_key == "custom_upload" else 1, horizontal=False)

st.sidebar.header("DATA")
if mode_choice == "Real Mission Data":
    st.sidebar.caption("Use imported mission products or custom archive data only when local products are available.")
else:
    st.sidebar.caption("Benchmark presets and synthetic scenarios remain clearly labeled as non-mission results.")

st.sidebar.header("CONFIGURATION")
with st.sidebar.expander("Advanced configuration", expanded=False):
    st.caption("Tune the registration engine before launching the workflow.")

st.sidebar.markdown("---")
st.sidebar.header("ABOUT")
st.sidebar.caption("PS 26166 · Chandrayaan-2 OHRC ↔ lunar reference imagery")
st.sidebar.caption("Supported missions: Chandrayaan-2, LRO, synthetic benchmark pairs")
if local_demo.get("summary"):
    st.sidebar.success(
        "Local evidence: "
        f"{local_demo['summary'].get('inlier_count', 0)} inliers | "
        f"RMSE {local_demo['summary'].get('rmse_pixels', 0.0):.3f}px"
    )
else:
    st.sidebar.info("No local evidence file detected yet; run the CLI benchmark to populate this panel.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset backend state", use_container_width=True):
    for key in [
        "result_report",
        "inliers",
        "raw_matches",
        "homography",
        "matcher_path",
        "warped_source",
        "img_source",
        "img_ref",
        "backend_trace",
        "last_run_summary",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

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

data_type_badge = (
    "REAL MISSION DATA"
    if st.session_state.get("mission_source_path")
    else data_type_label_for_key(selected_key)
)
summary_badge = "SYSTEM READY" if "result_report" not in st.session_state else "RESULT READY"
summary_badge = (
    f"EVIDENCE: RMSE {local_demo.get('summary', {}).get('rmse_pixels', 0.0):.3f}px"
    if local_demo.get("summary")
    else summary_badge
)

st.markdown(
    f"""
    <div class="portal-shell">
        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.75rem; letter-spacing:0.22em; text-transform:uppercase; color:#8eb8d8; margin-bottom:0.45rem;">SAMANVAYA</div>
                <h1 style="margin:0; font-size:2.15rem; color:#edf7ff;">Lunar Multi-Modal Image Correspondence</h1>
                <p style="margin:0.45rem 0 0; font-size:1.02rem; color:#b8d7f7;">Register Chandrayaan-2 and lunar reference imagery across illumination, viewpoint, and scale changes.</p>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:flex-end; margin-top:0.2rem;">
                <span style="padding:0.38rem 0.72rem; border-radius:999px; background: rgba(115, 200, 255, 0.12); border:1px solid rgba(115,200,255,0.35); color:#dff6ff; font-size:0.78rem; font-weight:700;">{data_type_badge}</span>
                <span style="padding:0.38rem 0.72rem; border-radius:999px; background: rgba(111, 227, 164, 0.12); border:1px solid rgba(111,227,164,0.35); color:#dfffea; font-size:0.78rem; font-weight:700;">{summary_badge}</span>
            </div>
        </div>
        <div style="margin-top:0.85rem; display:flex; flex-wrap:wrap; gap:0.6rem;">
            <span style="padding:0.34rem 0.72rem; border-radius:999px; background: rgba(255,255,255,0.02); border:1px solid rgba(148,176,230,0.22); color:#d9ebff; font-size:0.76rem; font-weight:600;">CPU/GPU: available</span>
            <span style="padding:0.34rem 0.72rem; border-radius:999px; background: rgba(255,255,255,0.02); border:1px solid rgba(148,176,230,0.22); color:#d9ebff; font-size:0.76rem; font-weight:600;">DATA STATUS: {'ready' if img_source is not None and img_ref is not None else 'pending'}</span>
            <span style="padding:0.34rem 0.72rem; border-radius:999px; background: rgba(255,255,255,0.02); border:1px solid rgba(148,176,230,0.22); color:#d9ebff; font-size:0.76rem; font-weight:600;">MODEL STATUS: active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if local_demo.get("summary"):
    st.caption(
        "Local benchmark evidence: "
        f"{local_demo['summary'].get('inlier_count', 0)} inliers, "
        f"{local_demo['summary'].get('inlier_ratio_percent', 0.0):.1f}% inlier ratio, "
        f"RMSE {local_demo['summary'].get('rmse_pixels', 0.0):.3f}px."
    )

st.sidebar.markdown("---")
st.sidebar.header("PAIR DISCOVERY")
st.sidebar.caption("Pair status remains conservative and reflects the actual backend state.")

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
uploaded_src_xml = None
uploaded_ref_xml = None

if selected_key == "custom_upload":
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Custom GeoTIFF Rasters")
    uploaded_src = st.sidebar.file_uploader("Upload Source GeoTIFF (OHRC / TMC-2)", type=["tif", "tiff", "geotiff", "png", "jpg"])
    uploaded_ref = st.sidebar.file_uploader("Upload Reference GeoTIFF (LRO NAC / Base)", type=["tif", "tiff", "geotiff", "png", "jpg"])
    uploaded_src_xml = st.sidebar.file_uploader("Upload Source PDS4 XML (Optional)", type=["xml"], key="source_pds4_xml")
    uploaded_ref_xml = st.sidebar.file_uploader("Upload Reference PDS4 XML (Optional)", type=["xml"], key="reference_pds4_xml")

local_mission_root = _PROJECT_ROOT / "data" / "raw" / "chandrayaan2"
local_product_records = scan_local_products_cached(str(local_mission_root))
st.sidebar.markdown("---")
st.sidebar.header("Latest available mission data")
st.sidebar.caption("Archive data is not a live camera feed. Archive reachability and local product availability are shown separately.")
if st.sidebar.button("Refresh official archive status", key="refresh_archive_status", width="stretch"):
    with st.sidebar.status("Checking official archive...", expanded=False) as archive_check_status:
        st.session_state["archive_status"] = check_archive()
        archive_check_status.update(label="Archive status checked", state="complete")
if st.sidebar.button("Scan local mission data", key="scan_local_mission_data", width="stretch"):
    scan_local_products_cached.clear()
    st.rerun()

local_products = [product_from_dict(record) for record in local_product_records]
selected_local_product = None
if local_products:
    instrument_options = ["All"] + sorted({product.instrument for product in local_products if product.instrument})
    selected_instrument = st.sidebar.selectbox("Instrument", instrument_options, key="mission_instrument_filter")
    visible_products = [
        product for product in local_products
        if selected_instrument == "All" or product.instrument == selected_instrument
    ]
    local_product_ids = [product.product_id or str(product.image_path) for product in visible_products]
    selected_product_id = st.sidebar.selectbox(
        "Authorized local product",
        ["None"] + local_product_ids,
        key="selected_local_product_id",
    )
    selected_local_product = next(
        (product for product in visible_products if (product.product_id or str(product.image_path)) == selected_product_id),
        None,
    )
    if selected_local_product is not None and (selected_local_product.band_count or 0) > 1:
        st.sidebar.warning("IIRS/spectral cube detected. Spectral-to-2-D registration preview is not implemented for this product.")
    if st.sidebar.button(
        "Use selected product as source",
        key="use_local_mission_source",
        width="stretch",
        disabled=selected_local_product is None or (selected_local_product.band_count or 0) > 1,
    ) and selected_local_product:
        st.session_state["mission_source_path"] = str(selected_local_product.image_path)
        clear_registration_state()
        st.rerun()
else:
    st.sidebar.caption(f"No authorized Chandrayaan-2 products found under {local_mission_root}.")

archive_status: Optional[MissionArchiveStatus] = st.session_state.get("archive_status")
if archive_status is None:
    archive_status = MissionArchiveStatus(
        source="ISRO / ISSDC",
        mission="Chandrayaan-2",
        last_checked="Not checked",
        latest_product_id=None,
        latest_acquisition_time=None,
        availability="NOT_CHECKED",
        access_mode="official archive; user authentication may be required",
        message="Refresh checks official archive reachability. It does not bypass login or download products.",
        official_url=OFFICIAL_ARCHIVES["Chandrayaan-2 PRADAN"],
    )
archive_status = status_with_local_products(archive_status, local_products)

st.markdown("## Latest available mission data")
status_col, action_col = st.columns([2, 1])
with status_col:
    st.caption("Truthful archive status for Chandrayaan-2. Product downloads remain user-authorized.")
    st.write(f"**{archive_status.availability}** · {archive_status.message}")
    st.write(f"Mission: {archive_status.mission} · Source: {archive_status.source}")
    st.write(f"Archive checked: {archive_status.last_checked}")
    st.write(f"Latest local product: {archive_status.latest_product_id or 'Not available'}")
    st.write(f"Mission acquired: {archive_status.latest_acquisition_time or 'Not available'}")
with action_col:
    st.link_button("Open official ISSDC archive", OFFICIAL_ARCHIVES["Chandrayaan-2 PRADAN"], width="stretch")
    st.link_button("Open Chandrayaan Data Explorer", OFFICIAL_ARCHIVES["Chandrayaan Data Explorer"], width="stretch")
    st.caption("Use the official authenticated portal to download products, then scan them locally.")

conf_thresh = st.session_state.get("ui_conf_thresh", 0.15)
anms_cap = st.session_state.get("ui_anms_cap", 4)
enable_subpixel = st.session_state.get("ui_enable_subpixel", True)
magsac_thresh = st.session_state.get("ui_magsac_thresh", 1.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Alignment Engine Parameters")
conf_thresh = st.sidebar.slider("LoFTR Confidence Threshold (τ)", 0.05, 0.90, conf_thresh, step=0.05, key="ui_conf_thresh")
anms_cap = st.sidebar.slider("ANMS Cap per Cell (8x8 Grid)", 1, 12, anms_cap, step=1, key="ui_anms_cap")
enable_subpixel = st.sidebar.checkbox("2D Parabolic Taylor Sub-Pixel Peak Refinement", value=enable_subpixel, key="ui_enable_subpixel")
magsac_thresh = st.sidebar.slider("USAC-MAGSAC++ Reprojection Threshold (px)", 0.5, 3.0, magsac_thresh, step=0.25, key="ui_magsac_thresh")

source_identity = getattr(uploaded_src, "name", "") if uploaded_src is not None else st.session_state.get("mission_source_path", selected_key)
reference_identity = getattr(uploaded_ref, "name", "") if uploaded_ref is not None else selected_key
registration_fingerprint = repr(
    (source_identity, reference_identity, selected_key, conf_thresh, anms_cap, enable_subpixel, magsac_thresh)
)
if st.session_state.get("registration_fingerprint") not in (None, registration_fingerprint):
    clear_registration_state()
st.session_state["registration_fingerprint"] = registration_fingerprint

# -----------------------------------------------------------------------------
# Ingestion & Data Preparation
# -----------------------------------------------------------------------------

if "backend_trace" not in st.session_state:
    st.session_state["backend_trace"] = [
        "Status: ready",
        "Backend: image ingestion and alignment pipeline idle",
        "Next action: choose benchmark and run alignment",
    ]

img_source: Optional[np.ndarray] = None
img_ref: Optional[np.ndarray] = None
sun_src: Optional[SunAngles] = None
sun_ref: Optional[SunAngles] = None
src_gsd = 1.0
ref_gsd = 1.0

if st.session_state.get("mission_source_path"):
    mission_source_path = Path(st.session_state["mission_source_path"])
    try:
        img_source = load_geotiff_file(mission_source_path)
        if selected_key in {"scenario_a", "scenario_b", "scenario_c"}:
            reference_benchmark = benchmarks.get(selected_key, {}).get("reference", {})
            reference_path = get_sample_data_dir() / reference_benchmark["filename"]
            img_ref = load_geotiff_file(reference_path)
        source_product = next(
            (product for product in local_products if str(product.image_path) == str(mission_source_path)),
            None,
        )
        if source_product is not None:
            src_gsd = source_product.gsd_m or 1.0
            if source_product.sun_azimuth_deg is not None and source_product.sun_elevation_deg is not None:
                sun_src = SunAngles(source_product.sun_azimuth_deg, source_product.sun_elevation_deg)
        data_type_badge = "REAL MISSION DATA"
        st.success(f"Loaded authorized local mission source: {source_product.product_id if source_product else mission_source_path.name}")
    except Exception as exc:
        st.error("MISSION PRODUCT COULD NOT BE LOADED")
        with st.expander("Technical details"):
            st.code(str(exc))
elif selected_key in ["scenario_a", "scenario_b", "scenario_c"]:
    bm = benchmarks.get(selected_key, {})
    sample_dir = get_sample_data_dir()
    src_path = sample_dir / bm["source"]["filename"]
    ref_path = sample_dir / bm["reference"]["filename"]

    try:
        img_source = load_geotiff_file(src_path)
        img_ref = load_geotiff_file(ref_path)
    except Exception as e:
        st.error(f"Failed to load cached benchmark GeoTIFF: {e}")

    st.markdown("<div style='height:0.38rem;'></div>", unsafe_allow_html=True)
    render_workflow_steps(1)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    left_col, center_col, right_col = st.columns([1, 1.15, 1])
    with left_col:
        render_pair_card(
            title="SOURCE",
            mission=format_unknown(bm.get('source', {}).get('spacecraft')),
            instrument=format_unknown(bm.get('source', {}).get('sensor')),
            product_id=format_unknown(bm.get('source', {}).get('product_id')), 
            dimensions=f"{format_unknown(bm.get('source', {}).get('width'))} × {format_unknown(bm.get('source', {}).get('height'))}",
            gsd=f"{format_unknown(bm.get('source', {}).get('gsd_m'))} m/px",
            status="READY",
        )
    with center_col:
        st.markdown(
            f"""
            <div style="padding:1rem 0.8rem; border:1px solid rgba(148,176,230,0.18); border-radius:16px; background:linear-gradient(180deg, rgba(12,21,30,0.96), rgba(10,18,27,0.94)); min-height:188px; display:flex; flex-direction:column; justify-content:center; text-align:center;">
                <div style="font-size:0.72rem; letter-spacing:0.14em; text-transform:uppercase; color:#8fb9d8; margin-bottom:0.6rem;">REGISTRATION CHALLENGE</div>
                <div style="font-size:1.05rem; color:#edfaff; font-weight:700;">GSD {bm.get('source', {}).get('gsd_m', 'Unknown')} → {bm.get('reference', {}).get('gsd_m', 'Unknown')} m/px</div>
                <div style="font-size:0.9rem; color:#d5ebff; margin-top:0.5rem;">Scale ratio: {format_unknown(bm.get('source', {}).get('gsd_m'))}</div>
                <div style="font-size:0.9rem; color:#d5ebff; margin-top:0.35rem;">Overlap: {format_unknown(bm.get('overlap_status', 'UNKNOWN'))}</div>
                <div style="font-size:0.9rem; color:#d5ebff; margin-top:0.35rem;">Geometry: {format_unknown(bm.get('geometry_method', 'unknown'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        render_pair_card(
            title="TARGET",
            mission=format_unknown(bm.get('reference', {}).get('spacecraft')),
            instrument=format_unknown(bm.get('reference', {}).get('sensor')),
            product_id=format_unknown(bm.get('reference', {}).get('product_id')),
            dimensions=f"{format_unknown(bm.get('reference', {}).get('width'))} × {format_unknown(bm.get('reference', {}).get('height'))}",
            gsd=f"{format_unknown(bm.get('reference', {}).get('gsd_m'))} m/px",
            status="READY",
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.caption(f"{bm.get('title', selected_benchmark)} · {bm.get('target', 'Moon')} · {bm.get('description', '')}")

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

            if uploaded_src_xml is not None:
                sun_src, src_gsd, src_sensor = parse_uploaded_pds4_metadata(uploaded_src_xml)
                st.info(
                    f"Source PDS4 metadata: {src_sensor.value}, {src_gsd:g} m/px, "
                    f"sun azimuth {sun_src.azimuth_deg:g}°, elevation {sun_src.elevation_deg:g}°."
                )
            if uploaded_ref_xml is not None:
                sun_ref, ref_gsd, ref_sensor = parse_uploaded_pds4_metadata(uploaded_ref_xml)
                st.info(
                    f"Reference PDS4 metadata: {ref_sensor.value}, {ref_gsd:g} m/px, "
                    f"sun azimuth {sun_ref.azimuth_deg:g}°, elevation {sun_ref.elevation_deg:g}°."
                )
        except Exception as e:
            st.error(f"Error reading custom imagery or PDS4 metadata: {e}")
    else:
        st.warning("Please upload both Source and Reference GeoTIFFs via the sidebar to execute registration.")

# -----------------------------------------------------------------------------
# Primary scientific workspace
# -----------------------------------------------------------------------------

st.markdown("## The registration workspace")
st.caption("The reference stays fixed. Samanvaya estimates a transform and aligns the moving source to it.")

if img_ref is None:
    st.info("Select a fixed reference image to begin.")
elif img_source is None:
    st.info("Select the moving Chandrayaan-2 image to begin.")
else:
    workspace_left, workspace_right = st.columns(2, gap="large")
    registered_preview = st.session_state.get("warped_source")
    with workspace_left:
        render_image_panel(
            "Reference frame",
            "FIXED IMAGE",
            img_ref,
            "Existing lunar reference used as the alignment target",
            data_type_badge,
        )
    with workspace_right:
        if registered_preview is not None:
            render_image_panel(
                "Registered warped source",
                "MOVING → ALIGNED TO REFERENCE",
                registered_preview,
                "Source after the computed geometric warp",
                data_type_badge,
            )
        else:
            render_image_panel(
                "Source image",
                "MOVING IMAGE",
                img_source,
                "Run registration to generate the aligned source",
                data_type_badge,
            )

    workflow_cols = st.columns(5)
    workflow_labels = (
        "01  Reference + source",
        "02  Find match points",
        "03  Estimate transform",
        "04  Warp source",
        "05  Verify alignment",
    )
    for column, label in zip(workflow_cols, workflow_labels):
        column.caption(label)

    with st.expander("Image metadata", expanded=False):
        metadata_left, metadata_right = st.columns(2)
        with metadata_left:
            st.markdown("**Reference frame · fixed**")
            st.write(f"Dimensions: {img_ref.shape[1]} × {img_ref.shape[0]} px")
            st.write(f"GSD: {ref_gsd:g} m/px" if ref_gsd is not None else "GSD: Not available")
            st.write("Sun geometry: available" if sun_ref is not None else "Sun geometry: unavailable")
        with metadata_right:
            st.markdown("**Source image · moving**")
            st.write(f"Dimensions: {img_source.shape[1]} × {img_source.shape[0]} px")
            st.write(f"GSD: {src_gsd:g} m/px" if src_gsd is not None else "GSD: Not available")
            st.write("Sun geometry: available" if sun_src is not None else "Sun geometry: unavailable")

# -----------------------------------------------------------------------------
# End-to-End Alignment Pipeline Execution with Progress Bar
# -----------------------------------------------------------------------------

if img_source is not None and img_ref is not None:
    col_btn, _ = st.columns([1, 3])
    run_alignment = col_btn.button("🚀 Execute Multi-Modal Alignment Pipeline", type="primary", use_container_width=True)

    if run_alignment:
        st.session_state["show_launch_modal"] = True
        st.session_state["launch_confirmed"] = False

    render_launch_modal(selected_benchmark, selected_key, source_modality, ref_modality)

    if st.session_state.get("launch_confirmed"):
        st.session_state["launch_confirmed"] = False
        st.session_state["show_launch_modal"] = False
        st.session_state["backend_trace"] = [
            "Status: running",
            "Backend: initializing planetary registration engine",
            "Stage: ingesting and normalizing source/reference imagery",
        ]

        progress_bar = st.progress(0, text="Initializing planetary registration engine...")

        # Step 1: Preprocessing & GeoTIFF normalizations
        progress_bar.progress(15, text="Step 1/5: Ingesting & normalizing GeoTIFF dynamic ranges...")
        st.session_state["backend_trace"] = [
            "Status: running",
            "Backend: image ingestion and dynamic-range normalization in progress",
            "Stage: loading source/reference rasters and metadata",
        ]
        time.sleep(0.1)

        # Step 2: Illumination-Invariant Log-Gabor Phase Congruency
        progress_bar.progress(35, text="Step 2/5: Vectorized 2D Log-Gabor Phase Congruency (PyTorch FFT)...")
        st.session_state["backend_trace"] = [
            "Status: running",
            "Backend: computing illumination-invariant phase congruency maps",
            "Stage: source/reference feature enhancement",
        ]
        pc_engine = PhaseCongruencyEngine(num_scales=4, num_orientations=6)
        pc_src_input = img_source
        pc_ref_input = img_ref
        if selected_key == "custom_upload" and sun_src is not None and sun_ref is not None:
            photometric = PhotometricNormalizer()
            pc_src_input, _ = photometric.normalize(img_source, sun_src, pixel_gsd=src_gsd)
            pc_ref_input, _ = photometric.normalize(img_ref, sun_ref, pixel_gsd=ref_gsd)
            st.caption("PDS4 solar geometry and GSD applied to custom-upload photometric preprocessing.")
        pc_src = pc_engine.compute(pc_src_input)
        pc_ref = pc_engine.compute(pc_ref_input)
        time.sleep(0.1)

        # Step 3: Dense LoFTR Cross-Attention Matching
        progress_bar.progress(60, text="Step 3/5: Dense Keypoint Extraction via kornia.feature.LoFTR...")
        st.session_state["backend_trace"] = [
            "Status: running",
            "Backend: dense matcher is extracting candidate correspondences",
            "Stage: LoFTR / RIFT feature matching and ANMS filtering",
        ]
        start_t = time.perf_counter()
        matcher = DenseLoFTRMatcher(
            pretrained="outdoor",
            confidence_threshold=conf_thresh,
            grid_bins=8,
            cap_per_cell=anms_cap,
            magsac_reproj_threshold=magsac_thresh,
        )
        if not matcher.is_pretrained:
            st.error("LoFTR pretrained weights are unavailable; results are not meaningful.")

        src_tensor, norm_src = matcher.prepare_geotiff_array(pc_src.max_moment)
        ref_tensor, norm_ref = matcher.prepare_geotiff_array(pc_ref.max_moment)

        raw_matches = matcher.extract_dense_correspondences(
            src_tensor, ref_tensor, norm_src.shape, norm_ref.shape
        )

        # Step 4: 8x8 Grid ANMS & 2D Parabolic Taylor Sub-pixel Refinement
        progress_bar.progress(80, text="Step 4/5: Grid-Based ANMS (8x8 Grid) & 2D Parabolic Taylor Refinement...")
        anms_matches = matcher.apply_grid_anms_8x8(raw_matches, norm_src.shape, use_source_coords=True)

        if enable_subpixel and anms_matches:
            refined_matches = matcher.refine_subpixel_taylor_2d(anms_matches, norm_src, norm_ref)
        else:
            refined_matches = anms_matches

        # Step 5: USAC-MAGSAC Homography & Warping
        progress_bar.progress(95, text="Step 5/5: USAC-MAGSAC++ Homography Estimation & Warping...")
        st.session_state["backend_trace"] = [
            "Status: running",
            "Backend: filtering outliers and estimating alignment transform",
            "Stage: robust MAGSAC homography and warping",
        ]
        inliers, H, warped_source = matcher.filter_outliers_magsac(
            refined_matches, img_source, img_ref.shape
        )
        matcher_path = "dense_loftr"
        if len(inliers) < 4:
            matcher_path = "classical_rift"
            rift_matches = ClassicalRIFTMatcher().match(
                pc_ref.max_moment,
                pc_ref.orientation_max_idx,
                pc_src.max_moment,
                pc_src.orientation_max_idx,
            )
            inliers, H, warped_source = matcher.filter_outliers_magsac(
                rift_matches, img_source, img_ref.shape
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
        st.session_state["backend_trace"] = [
            "Status: complete",
            f"Backend: alignment finished in {elapsed_ms:.1f} ms",
            f"Stage: validation summary -> RMSE {report.rmse_pixels:.4f}px, inliers {report.inlier_count}",
        ]
        st.session_state["last_run_summary"] = {
            "matcher": matcher_path,
            "rmse_px": float(report.rmse_pixels),
            "inlier_count": int(report.inlier_count),
            "ground_truth_available": bool(report.ground_truth_available),
            "meets_isro_mandate": bool(report.meets_isro_mandate),
            "processing_time_ms": float(report.processing_time_ms),
        }
        time.sleep(0.2)
        progress_bar.empty()

        st.session_state["result_report"] = report
        st.session_state["inliers"] = inliers
        st.session_state["raw_matches"] = raw_matches
        st.session_state["homography"] = H
        st.session_state["matcher_path"] = matcher_path
        st.session_state["warped_source"] = warped_source
        st.session_state["img_source"] = img_source
        st.session_state["img_ref"] = img_ref

# -----------------------------------------------------------------------------
# Results Presentation: Scorecards, Plots & Blending
# -----------------------------------------------------------------------------

st.markdown("---")
backend_trace = st.session_state.get("backend_trace", ["Status: ready", "Backend: idle", "Next action: choose benchmark and run alignment"])
backend_status = backend_trace[0]

status_container = st.container()
with status_container:
    st.markdown(
        f"<div class='metric-card'><div class='status-badge'>{backend_status}</div>"
        f"<div style='margin-top: 0.7rem; color: #d9ebff; font-size:1rem;'>"
        f"<strong>Backend activity:</strong> {' · '.join(backend_trace[1:])}</div></div>",
        unsafe_allow_html=True,
    )

if "result_report" in st.session_state:
    report: RegistrationEvaluationReport = st.session_state["result_report"]
    inliers = st.session_state["inliers"]
    H = st.session_state["homography"]
    matcher_path = st.session_state.get("matcher_path", "dense_loftr")
    warped_src = st.session_state["warped_source"]
    img_source = st.session_state["img_source"]
    img_ref = st.session_state["img_ref"]

    st.markdown("---")
    st.caption(f"Matcher path: {matcher_path}")
    st.subheader("Registration result")
    result_status = "SUCCESS" if report.inlier_count >= 4 and H is not None else "LOW CONFIDENCE"
    if report.inlier_count == 0 or H is None:
        result_status = "FAILED"
    st.markdown(f"**Status: {result_status}** · {data_type_badge}")

    # Every metric below is produced by the evaluation backend.
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    raw_match_count = len(st.session_state.get("raw_matches", []))
    col_k1.metric(
        "Reprojection RMSE",
        f"{report.rmse_pixels:.3f} px",
        delta=f"{report.rmse_pixels - 0.40:.3f} vs 0.40 px target",
        delta_color="inverse",
    )
    col_k2.metric("Raw matches", f"{raw_match_count} pts")
    col_k3.metric("Inliers", f"{report.inlier_count} pts")
    col_k4.metric("Inlier ratio", f"{report.inlier_ratio_percent:.1f}%")

    col_k5, col_k6, col_k7, col_k8 = st.columns(4)
    col_k5.metric("Spatial coverage", f"{report.spatial_uniformity_entropy:.3f} / 1.0")
    col_k6.metric("Runtime", f"{report.processing_time_ms:.1f} ms")
    col_k7.metric(
        "Ground-truth RMSE",
        f"{report.control_point_rmse_pixels:.3f} px" if report.control_point_rmse_pixels is not None else "Not available",
    )
    col_k8.metric("Tie points", f"{len(report.tie_points)}")

    with st.expander("How the source was aligned", expanded=False):
        st.write("Source: Moving image")
        st.write("Reference: Fixed image")
        st.write(f"Transformation matrix: {np.asarray(H).tolist() if H is not None else 'Not available'}")
        st.write(f"Matcher: {matcher_path}")
        st.write(f"Subpixel refinement: {'Applied' if any(pt.get('subpixel_refined', False) for pt in report.tie_points) else 'Not available'}")
        st.write(f"Fallback: {'Classical RIFT' if matcher_path == 'classical_rift' else 'Not used'}")

    if report.ground_truth_available and report.meets_isro_mandate:
        st.success("🎯 **Independent ground-truth validation passed**: RMSE < 0.40 px and sufficient inliers were confirmed for this run.")
    elif report.ground_truth_available:
        st.warning("⚠️ Independent ground truth is available, but the run did not satisfy the validation threshold. Review the residuals and matcher configuration.")
    else:
        st.warning("⚠️ This run is based on reprojection consensus only. No independent ground truth was supplied, so it cannot be claimed as scientific validation.")

    st.subheader("🧠 Backend execution trace")
    st.code("\n".join(backend_trace), language="text")

    # Interactive inspection tabs use only backend-produced imagery and points.
    tab_source, tab_reference, tab_registered, tab_overlap, tab_tiepoints, tab_diagnostics, tab_exports = st.tabs([
        "Source",
        "Reference",
        "Registered",
        "Overlay",
        "Match points",
        "Residuals",
        "Exports",
    ])

    with tab_source:
        render_image_panel(
            "Source image",
            "MOVING IMAGE",
            img_source,
            "Original source before geometric alignment",
            data_type_badge,
        )

    with tab_reference:
        render_image_panel(
            "Reference frame",
            "FIXED IMAGE",
            img_ref,
            "Fixed image used as the registration target",
            data_type_badge,
        )

    with tab_registered:
        if warped_src is None:
            st.info("Registration not yet run.")
        else:
            render_image_panel(
                "Registered warped source",
                "MOVING → ALIGNED TO REFERENCE",
                warped_src,
                "Actual warped source produced by the registration backend",
                data_type_badge,
            )

    # TAB 1: 50/50 Checkerboard Blend and Sliding Wipe Tool
    with tab_overlap:
        st.markdown("### Reference + registered source")
        st.caption("Adjust opacity to inspect whether crater rims, ridges, and boundaries coincide.")
        if warped_src is not None:
            opacity = st.slider("Registered source opacity", 0, 100, 50, key="registered_opacity")
            overlay = np.clip((1.0 - opacity / 100.0) * img_ref + (opacity / 100.0) * warped_src, 0.0, 1.0)
            st.image(moon_preview_image(overlay), caption=f"Reference {100 - opacity}% · Registered source {opacity}%", width="stretch")
        else:
            st.info("Run registration to generate the registered-source overlay.")

        st.markdown("### Before → after")
        st.caption("Drag the wipe slider from 0% to 100% across the boundary. Continuous crater boundaries confirm sub-pixel planetary alignment.")

        if warped_src is not None:
            split_pct = st.slider("Sliding Wipe Divider Position (%)", 0, 100, 50, step=1)
            h, w = img_ref.shape
            col_split = int((split_pct / 100.0) * w)

            swipe_composite = np.zeros_like(img_ref)
            swipe_composite[:, :col_split] = img_ref[:, :col_split]
            swipe_composite[:, col_split:] = warped_src[:, col_split:]

            swipe_rgb = moon_preview_image(swipe_composite)
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

            st.image(moon_preview_image(checker), caption=f"50/50 Checkerboard Blend ({tile_size}x{tile_size} px tiles)", width="stretch", clamp=True)

    # TAB 2: Interactive Side-by-Side Tie-Point Correspondence Plot
    with tab_tiepoints:
        st.subheader("Match points")
        st.caption("Match points are common lunar landmarks detected in both images, such as crater rims, ridges, or other distinctive surface features.")
        st.caption(f"Raw matches: {len(st.session_state.get('raw_matches', []))} · Filtered/inliers shown below: {len(inliers)}")

        max_pts = st.slider("Max Tie-Points to Display", 10, 150, 40, step=5)
        if inliers:
            fig_corr = render_tie_point_correspondences(img_ref, img_source, inliers, max_display=max_pts)
            st.pyplot(fig_corr)
        else:
            st.info("No reliable correspondence points were found.")

    # TAB 3: Residual Error Scatter & Frequency Invariance
    with tab_diagnostics:
        st.subheader("📈 Residual Error Scatter Field & Frequency Invariance")
        st.markdown("#### Inlier Vector Flow Field")
        st.caption("Displacement vectors show the direction and magnitude of each inlier match.")
        if inliers:
            ref_x = np.array([match.ref_xy[0] for match in inliers])
            ref_y = np.array([match.ref_xy[1] for match in inliers])
            delta_x = np.array([match.target_xy[0] - match.ref_xy[0] for match in inliers])
            delta_y = np.array([match.target_xy[1] - match.ref_xy[1] for match in inliers])
            magnitudes = np.hypot(delta_x, delta_y)

            flow_fig, flow_ax = plt.subplots(figsize=(10, 7))
            flow_ax.imshow(moon_preview_image(img_ref), interpolation="nearest")
            flow = flow_ax.quiver(
                ref_x, ref_y, delta_x, delta_y, magnitudes,
                cmap="autumn", angles="xy", scale_units="xy", scale=1.0,
                width=0.005,
            )
            flow_ax.scatter(ref_x, ref_y, c="lime", s=25, edgecolors="black", linewidths=0.5)
            flow_ax.set_title(f"Inlier Vector Flow Field ({len(inliers)} matches)")
            flow_ax.axis("off")
            flow_fig.colorbar(flow, ax=flow_ax, shrink=0.7, label="Displacement Magnitude (pixels)")
            st.pyplot(flow_fig)
        else:
            st.warning("No inliers detected for vector flow field.")

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
            st.image(moon_preview_image(pc_out.max_moment), caption="Maximum Moment (M_max) Invariant Step Edges", width="stretch", clamp=True)

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
else:
    st.info("The backend is idle. Choose a benchmark and run the alignment pipeline to populate live metrics and validation output.")
    st.code("\n".join(backend_trace), language="text")
