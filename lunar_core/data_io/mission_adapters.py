"""Mission-specific views over the common product catalog."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from lunar_core.data_io.mission_catalog import scan
from lunar_core.data_io.mission_product import MissionProduct


class MissionAdapter(ABC):
    """Common adapter contract for metadata-only mission discovery."""

    mission_name: str
    instruments: tuple[str, ...] = ()
    identifier_tokens: tuple[str, ...] = ()
    instrument_tokens: dict[str, tuple[str, ...]] = {}

    @abstractmethod
    def scan(self, root: Path) -> list[MissionProduct]:
        """Discover products belonging to this mission."""

    def _filter(self, root: Path) -> list[MissionProduct]:
        products = scan(root)
        selected: list[MissionProduct] = []
        for product in products:
            self._complete_identity(product)
            if product.mission == self.mission_name and (
                not self.instruments or product.instrument in self.instruments
            ):
                selected.append(product)
        return selected

    def _complete_identity(self, product: MissionProduct) -> None:
        """Use mission-specific identifiers only when generic metadata is absent."""
        if product.status != "validated":
            return
        identifier = " ".join(
            value
            for value in (product.product_id, product.image_path.stem if product.image_path else None)
            if value
        ).upper()
        if product.mission is None and not any(token in identifier for token in self.identifier_tokens):
            return
        if product.mission is not None and product.mission != self.mission_name:
            return

        if product.mission is None:
            product.mission = self.mission_name
        if product.instrument is None:
            for instrument, tokens in self.instrument_tokens.items():
                if any(token in identifier for token in tokens):
                    product.instrument = instrument
                    break
        if product.identification_method is None:
            product.identification_method = "mission_adapter_heuristic"


class Chandrayaan2Adapter(MissionAdapter):
    mission_name = "Chandrayaan-2"
    instruments = ("OHRC", "TMC-2", "IIRS")
    identifier_tokens = ("CH2", "CHANDRAYAAN-2", "CHANDRAYAAN2")
    instrument_tokens = {
        "OHRC": ("OHRC", "OHR"),
        "TMC-2": ("TMC", "TMC2"),
        "IIRS": ("IIRS",),
    }

    def scan(self, root: Path) -> list[MissionProduct]:
        return self._filter(root)


class LROAdapter(MissionAdapter):
    mission_name = "LRO"
    instruments = ("NAC", "LROC")
    identifier_tokens = ("LRO", "LROC", "NAC")
    instrument_tokens = {
        "NAC": ("NAC",),
        "LROC": ("LROC",),
    }

    def scan(self, root: Path) -> list[MissionProduct]:
        return self._filter(root)


class SeleneAdapter(MissionAdapter):
    mission_name = "SELENE"
    instruments = ("TC",)
    identifier_tokens = ("SELENE", "KAGUYA")
    instrument_tokens = {"TC": ("TC", "TERRAIN CAMERA")}

    def scan(self, root: Path) -> list[MissionProduct]:
        return self._filter(root)


ADAPTERS = {
    "chandrayaan2": Chandrayaan2Adapter,
    "lro": LROAdapter,
    "selene": SeleneAdapter,
}


def adapter_for_mission(mission: str) -> Optional[MissionAdapter]:
    adapter_type = ADAPTERS.get(mission.lower().replace("-", ""))
    return adapter_type() if adapter_type else None
