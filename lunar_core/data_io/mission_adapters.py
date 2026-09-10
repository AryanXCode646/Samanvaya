"""Mission-specific metadata views over the common product catalog.

These adapters currently normalize and filter catalog products. They are not
mission-native calibration or instrument-cube decoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from lunar_core.data_io.mission_catalog import scan
from lunar_core.data_io.mission_product import IdentificationMethod, MissionProduct
from lunar_core.data_io.product_identity import identify_from_product_tokens


class MissionAdapter(ABC):
    """Common adapter contract for metadata-only mission discovery."""

    mission_name: str
    instruments: tuple[str, ...] = ()

    @property
    def supported_instruments(self) -> tuple[str, ...]:
        return self.instruments

    @classmethod
    def canonicalize_mission_name(cls, value: str) -> str:
        cleaned = (value or "").strip().lower().replace("-", "").replace("_", "")
        mapping = {
            "chandrayaan2": "Chandrayaan-2",
            "ch2": "Chandrayaan-2",
            "chandrayaan": "Chandrayaan-2",
            "lro": "LRO",
            "selene": "SELENE",
            "kaguya": "SELENE",
        }
        return mapping.get(cleaned, value.strip())

    def matches_mission(self, mission: Optional[str]) -> bool:
        if mission is None:
            return False
        return self.canonicalize_mission_name(mission) == self.mission_name

    def matches_instrument(self, instrument: Optional[str]) -> bool:
        if instrument is None:
            return False
        normalized = instrument.strip().upper()
        aliases = {"OHR": "OHRC", "TMC": "TMC-2", "TMC2": "TMC-2", "LROC_NAC": "NAC", "NAC": "NAC", "LROCNAC": "NAC"}
        normalized = aliases.get(normalized, normalized)
        return normalized in {value.upper() for value in self.supported_instruments}

    @abstractmethod
    def scan(self, root: Path) -> list[MissionProduct]:
        """Discover products belonging to this mission."""

    def _filter(self, root: Path) -> list[MissionProduct]:
        products = scan(root)
        selected: list[MissionProduct] = []
        for product in products:
            self._complete_identity(product)
            if self.matches_mission(product.mission) and (
                not self.instruments or self.matches_instrument(product.instrument)
            ):
                selected.append(product)
        return selected

    def _complete_identity(self, product: MissionProduct) -> None:
        """Fill missing identity only from constrained mission-specific identifiers."""
        if product.status != "validated":
            return
        if product.mission is not None and product.instrument is not None:
            return
        image_stem = product.image_path.stem if product.image_path else None
        mission, instrument = identify_from_product_tokens(product.product_id, image_stem)
        if mission is not None and product.mission is None:
            if self.canonicalize_mission_name(mission) != self.mission_name:
                return
            product.mission = mission
            product.identification_method = IdentificationMethod.MISSION_SPECIFIC_IDENTIFIER.value
        if instrument is not None and product.instrument is None:
            product.instrument = instrument
            product.identification_method = IdentificationMethod.MISSION_SPECIFIC_IDENTIFIER.value


class Chandrayaan2Adapter(MissionAdapter):
    mission_name = "Chandrayaan-2"
    instruments = ("OHRC", "TMC-2", "IIRS")

    def scan(self, root: Path) -> list[MissionProduct]:
        return self._filter(root)


class LROAdapter(MissionAdapter):
    mission_name = "LRO"
    instruments = ("NAC", "LROC")

    def scan(self, root: Path) -> list[MissionProduct]:
        return self._filter(root)


class SeleneAdapter(MissionAdapter):
    mission_name = "SELENE"
    instruments = ("TC",)

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
