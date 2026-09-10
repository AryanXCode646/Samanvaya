"""lunar_core package.

The package exposes a few high-level model types at the top level, but importing the
full pipeline eagerly pulls in optional runtime dependencies such as torch and OpenCV.
That makes lightweight metadata-only workflows like mission catalog scanning fail
during import even when the caller does not need the full pipeline.
"""

from typing import Any

from lunar_core.models import (
    GeoRaster,
    SunAngles,
    KeypointMatch,
    RegistrationMetrics,
    RegistrationResult,
    SensorModality,
    TransformationType,
)

__version__ = "1.0.0"
__all__ = [
    "GeoRaster",
    "SunAngles",
    "KeypointMatch",
    "RegistrationMetrics",
    "RegistrationResult",
    "SensorModality",
    "TransformationType",
    "LunarCorePipeline",
]


def __getattr__(name: str) -> Any:
    if name == "LunarCorePipeline":
        from lunar_core.pipeline import LunarCorePipeline

        return LunarCorePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
