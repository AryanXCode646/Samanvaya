from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.data_io.product_identity import identify_from_product_tokens
from lunar_core.data_io.raster_reader import sanitize_path

DEFAULT_MANIFEST_PATH = Path("data/real/manifest.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/real_validation")


def _artifact_root_for_manifest(manifest_path: str | Path, artifact_root: str | Path | None = None) -> Path:
    if artifact_root is not None:
        return Path(artifact_root).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    return manifest_file.parent / "artifacts" / "real_validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_pair_id(source_path: Path, target_path: Path) -> str:
    source_token = source_path.stem.upper().replace("-", "_").replace(" ", "_")
    target_token = target_path.stem.upper().replace("-", "_").replace(" ", "_")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"PAIR_{source_token}_{target_token}_{now}"


def _normalize_product(product: Any) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "mission": product.mission,
        "instrument": product.instrument,
        "image_path": str(product.image_path),
        "label_path": str(product.label_path) if product.label_path else None,
        "width": product.width,
        "height": product.height,
        "band_count": product.band_count,
        "gsd_m": product.gsd_m,
        "acquisition_time": product.acquisition_time,
        "center_lat_deg": product.center_lat_deg,
        "center_lon_deg": product.center_lon_deg,
        "status": str(product.status),
        "validation_status": product.validation_status,
        "validation_message": product.validation_message,
        "geometry_method": product.geometry_method,
        "footprint": product.footprint,
    }


def _manifest_path_for(value: Optional[str | Path]) -> Path:
    if value is None:
        return DEFAULT_MANIFEST_PATH
    path = Path(value).expanduser().resolve()
    if path.suffix == "":
        path = path / "manifest.json"
    return path


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(), "pairs": []}


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _product_suitable_for_real_pair(product: Any) -> bool:
    if product is None:
        return False
    if getattr(product, "status", "") in {"invalid", "unsupported"}:
        return False

    instrument = str(getattr(product, "instrument", "") or "").upper()
    if instrument in {"OHRC", "NAC", "LROC", "TMC-2", "IIRS"}:
        return True

    mission, inferred_instrument = identify_from_product_tokens(
        getattr(product, "product_id", None),
        getattr(getattr(product, "image_path", None), "stem", None),
    )
    if mission is not None and inferred_instrument is not None:
        return True
    if inferred_instrument is not None:
        return inferred_instrument.upper() in {"OHRC", "NAC", "LROC", "TMC-2", "IIRS"}
    return False


def import_real_pair(
    ohrc_path: str | Path,
    lroc_path: str | Path,
    *,
    pair_id: Optional[str] = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> str:
    source_path = sanitize_path(ohrc_path)
    if not source_path.exists():
        raise FileNotFoundError("OHRC product not found: {source_path}")

    target_path = sanitize_path(lroc_path)
    if not target_path.exists():
        raise FileNotFoundError(
            "LROC NAC product not found.\n\n"
            "OHRC:\n"
            "  VERIFIED\n\n"
            "LROC NAC:\n"
            "  MISSING\n\n"
            "No registration was executed.\n"
            "No metrics were fabricated."
        )

    source_product = inspect_product(source_path, root_dir=source_path.parent)
    target_product = inspect_product(target_path, root_dir=target_path.parent)

    if not _product_suitable_for_real_pair(source_product):
        raise ValueError(f"Invalid OHRC product: {source_path}")
    if not _product_suitable_for_real_pair(target_product):
        raise ValueError(f"Invalid LROC product: {target_path}")

    source_mission, source_instrument = identify_from_product_tokens(
        getattr(source_product, "product_id", None),
        getattr(getattr(source_product, "image_path", None), "stem", None),
    )
    target_mission, target_instrument = identify_from_product_tokens(
        getattr(target_product, "product_id", None),
        getattr(getattr(target_product, "image_path", None), "stem", None),
    )
    source_mission = str(source_mission or getattr(source_product, "mission", "") or "").upper()
    source_instrument = str(source_instrument or getattr(source_product, "instrument", "") or "").upper()
    target_mission = str(target_mission or getattr(target_product, "mission", "") or "").upper()
    target_instrument = str(target_instrument or getattr(target_product, "instrument", "") or "").upper()
    if "CHANDRAYAAN" not in source_mission and "OHRC" not in source_instrument:
        raise ValueError(f"Could not identify OHRC mission/instrument metadata in {source_path}")
    if "LRO" not in target_mission and "NAC" not in target_instrument and "LROC" not in target_instrument:
        raise ValueError(f"Could not identify LROC NAC mission/instrument metadata in {target_path}")

    final_pair_id = pair_id or _canonical_pair_id(source_path, target_path)
    source_meta = _normalize_product(source_product)
    target_meta = _normalize_product(target_product)
    source_meta["sha256"] = _sha256(source_path)
    target_meta["sha256"] = _sha256(target_path)
    source_meta["preserved_original_path"] = str(source_path)
    target_meta["preserved_original_path"] = str(target_path)

    manifest_path = Path(manifest_path).expanduser().resolve()
    artifact_root_path = _artifact_root_for_manifest(manifest_path, artifact_root)
    manifest = _load_manifest(manifest_path)
    entry = {
        "pair_id": final_pair_id,
        "mission_pair": "Chandrayaan-2 OHRC ↔ LROC NAC",
        "source_product": source_meta,
        "reference_product": target_meta,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "imported",
        "validation_status": "pending",
        "ground_truth_status": "PENDING_INDEPENDENT_GROUND_TRUTH",
        "artifacts_dir": str(artifact_root_path / final_pair_id),
        "source_sha256": source_meta["sha256"],
        "reference_sha256": target_meta["sha256"],
        "provenance": {
            "source_file": str(source_path),
            "reference_file": str(target_path),
            "import_method": "repository_real_pair_import",
            "imported_by": "samanvaya.data_io.import_real_pair",
        },
    }
    manifest.setdefault("pairs", []).append(entry)
    _save_manifest(Path(manifest_path), manifest)

    artifact_dir = artifact_root_path / final_pair_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return final_pair_id


def _main() -> None:
    parser = argparse.ArgumentParser(description="Import a real Chandrayaan-2 OHRC and LROC NAC pair into the Samanvaya manifest.")
    parser.add_argument("--ohrc", required=True, help="Path to the real OHRC product")
    parser.add_argument("--lroc", required=True, help="Path to the real LROC NAC product")
    parser.add_argument("--pair-id", default=None, help="Optional explicit pair ID")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to the manifest JSON file")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT), help="Root directory for pair artifacts")
    args = parser.parse_args()
    pair_id = import_real_pair(args.ohrc, args.lroc, pair_id=args.pair_id, manifest_path=args.manifest, artifact_root=args.artifact_root)
    print(f"Imported pair {pair_id}")


if __name__ == "__main__":
    _main()
