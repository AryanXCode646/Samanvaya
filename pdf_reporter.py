#!/usr/bin/env python3
"""
Samanvaya (समान्वय) — Automated Executive PDF Mission Report Generator
ISRO SIH PS 26166: Multi-Modal Lunar Optical Image Registration Framework

Thin CLI entrypoint wrapping lunar_core.evaluation.pdf_reporter.
"""

from __future__ import annotations

from lunar_core.evaluation.pdf_reporter import (
    MissionReportGenerator,
    SamanvayaMissionReportGenerator,
    main,
)

__all__ = [
    "MissionReportGenerator",
    "SamanvayaMissionReportGenerator",
    "main",
]

if __name__ == "__main__":
    main()
