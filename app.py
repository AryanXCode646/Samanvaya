"""
app.py - Root Streamlit Entrypoint for Samanvaya (समान्वय).
ISRO Chandrayaan-2 Planetary Image Registration Portal (SIH PS 26166).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Launch interactive Streamlit UI
import runpy

runpy.run_module("lunar_core.ui.app", run_name="__main__")
