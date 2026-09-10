"""Data I/O helpers.

Keep the package lightweight to support metadata-only workflows. Heavy processing
modules like the tile processor pull in optional deep-learning dependencies and
should be loaded only when explicitly requested.
"""

from typing import Any

__all__ = [
    "PlanetaryRasterReader",
    "PlanetaryRasterWriter",
    "PlanetaryTileProcessor",
    "TileProcessingResult",
    "MissionProduct",
    "MissionAdapter",
    "Chandrayaan2Adapter",
    "LROAdapter",
    "SeleneAdapter",
    "ProductPair",
    "propose_pair",
    "inspect_product",
    "scan",
]


def __getattr__(name: str) -> Any:
    if name == "PlanetaryRasterReader":
        from lunar_core.data_io.raster_reader import PlanetaryRasterReader

        return PlanetaryRasterReader
    if name == "PlanetaryRasterWriter":
        from lunar_core.data_io.raster_writer import PlanetaryRasterWriter

        return PlanetaryRasterWriter
    if name in {"PlanetaryTileProcessor", "TileProcessingResult"}:
        from lunar_core.data_io.tile_processor import PlanetaryTileProcessor, TileProcessingResult

        if name == "PlanetaryTileProcessor":
            return PlanetaryTileProcessor
        return TileProcessingResult
    if name == "MissionProduct":
        from lunar_core.data_io.mission_product import MissionProduct

        return MissionProduct
    if name == "inspect_product":
        from lunar_core.data_io.mission_catalog import inspect_product

        return inspect_product
    if name == "scan":
        from lunar_core.data_io.mission_catalog import scan

        return scan
    if name in {"MissionAdapter", "Chandrayaan2Adapter", "LROAdapter", "SeleneAdapter"}:
        from lunar_core.data_io.mission_adapters import (
            Chandrayaan2Adapter,
            LROAdapter,
            MissionAdapter,
            SeleneAdapter,
        )

        mapping = {
            "MissionAdapter": MissionAdapter,
            "Chandrayaan2Adapter": Chandrayaan2Adapter,
            "LROAdapter": LROAdapter,
            "SeleneAdapter": SeleneAdapter,
        }
        return mapping[name]
    if name in {"ProductPair", "propose_pair"}:
        from lunar_core.data_io.pair_selector import ProductPair, propose_pair

        if name == "ProductPair":
            return ProductPair
        return propose_pair
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
