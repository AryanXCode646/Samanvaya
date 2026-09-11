"""Evidence-first discovery APIs for supplied mission and reference products."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lunar_core.data_io.mission_adapters import LROAdapter, SeleneAdapter, Chandrayaan2Adapter
from lunar_core.data_io.mission_catalog import scan
from lunar_core.data_io.mission_product import MissionProduct
from lunar_core.data_io.pair_selector import ProductPair, propose_pair


def _status_value(product: MissionProduct) -> str:
    return str(getattr(product.status, "value", product.status))


def classify_product(product: MissionProduct) -> str:
    """Classify product into AUTHORIZED_REAL, SYNTHETIC_FIXTURE, UNVERIFIED, or INVALID."""
    status = _status_value(product).lower()

    # 1. Explicit invalid detection (priority over fixture tags)
    if status in {"invalid", "unsupported"}:
        return "INVALID"
    if product.validation_message and ("fail" in str(product.validation_message).lower() or "error" in str(product.validation_message).lower()):
        return "INVALID"
    if product.width is not None and product.height is not None:
        if product.width <= 0 or product.height <= 0:
            return "INVALID"

    # 2. Synthetic fixture detection (scoped to fixture directory names, not pytest temp paths)
    img_path = Path(product.image_path) if product.image_path else Path()
    path_parts = {part.lower() for part in img_path.parts}
    prod_id = str(product.product_id or "").lower()
    synthetic_markers = {"synthetic", "fixture", "fixtures", "sample_data", "mock", "dummy"}
    if bool(path_parts & synthetic_markers) or any(m in prod_id for m in synthetic_markers) or img_path.stem.lower().startswith("synthetic") or img_path.stem.lower().startswith("synth"):
        return "SYNTHETIC_FIXTURE"

    # 3. Verify existence on filesystem
    if not product.image_path or not img_path.is_file():
        return "UNVERIFIED"

    # Recognized planetary missions
    mission_str = str(product.mission or "").upper()
    valid_missions = ("CHANDRAYAAN-2", "CHANDRAYAAN-1", "CH2", "CH1", "LRO", "LROC", "SELENE", "KAGUYA")
    if not any(vm in mission_str for vm in valid_missions):
        return "UNVERIFIED"

    # Verified flight product with validated metadata
    if status == "validated" and product.gsd_m is not None and product.gsd_m > 0 and product.width and product.height:
        return "AUTHORIZED_REAL"

    return "UNVERIFIED"


def product_inventory_record(product: MissionProduct) -> dict[str, object]:
    record = product.to_dict()
    required = ("mission", "instrument", "product_id", "width", "height", "band_count", "gsd_m", "acquisition_time")
    present = sum(record.get(field) not in (None, "") for field in required)
    record["metadata_completeness"] = f"{present}/{len(required)}"
    record["path"] = record.get("image_path")
    record["classification"] = classify_product(product)
    return record


def inventory_products(root: str | Path) -> list[dict[str, object]]:
    return [product_inventory_record(product) for product in discover_products(root)]


def discover_products(root: str | Path) -> list[MissionProduct]:
    """Recursively inspect candidate rasters and retain explicit failure statuses."""
    return scan(Path(root))


def discover_mission_products(root: str | Path, mission: Optional[str] = None) -> list[MissionProduct]:
    """Discover products through a constrained mission adapter."""
    if mission is None:
        return discover_products(root)
    normalized = mission.lower().replace("-", "")
    adapter = {
        "chandrayaan2": Chandrayaan2Adapter,
        "lro": LROAdapter,
        "selene": SeleneAdapter,
    }.get(normalized)
    if adapter is None:
        raise ValueError(f"Unsupported mission adapter: {mission}")
    return adapter().scan(Path(root))


def discover_reference_products(root: str | Path) -> list[MissionProduct]:
    """Discover supported lunar reference products without mixing mission classes."""
    return LROAdapter().scan(Path(root)) + SeleneAdapter().scan(Path(root))


def discover_benchmark_pairs(
    root: str | Path,
    *,
    source_mission: Optional[str] = None,
    reference_only: bool = False,
) -> list[ProductPair]:
    """Generate conservative candidate pairs with diagnostic selection reasons."""
    products = discover_products(root)
    references = [product for product in products if product.mission in {"LRO", "SELENE"} and _status_value(product) in {"validated", "partial"}]
    sources = [product for product in products if product not in references and _status_value(product) in {"validated", "partial"}]
    if source_mission:
        sources = [product for product in sources if product.mission == source_mission]
    if reference_only:
        sources = references
        references = []
    return [propose_pair(source, reference) for source in sources for reference in references]
