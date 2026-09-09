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

    @abstractmethod
    def scan(self, root: Path) -> list[MissionProduct]:
        """Discover products belonging to this mission."""

    def _filter(self, root: Path) -> list[MissionProduct]:
        products = scan(root)
        return [
            product
            for product in products
            if product.mission == self.mission_name
            and (not self.instruments or product.instrument in self.instruments)
        ]


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
