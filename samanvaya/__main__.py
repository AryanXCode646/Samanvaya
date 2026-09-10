from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.data_io.raster_reader import sanitize_path


def _iter_candidate_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".img", ".tif", ".tiff", ".dat"}:
            yield path


def _normalize_status(value: Any) -> str:
    text = str(value).lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def discover_real_data(root: str | Path) -> dict[str, Any]:
    root_path = sanitize_path(root)
    seen: list[Any] = []
    for image_path in _iter_candidate_images(root_path):
        try:
            product = inspect_product(image_path, root_dir=root_path)
        except Exception:
            continue
        if getattr(product, "product_id", None) is not None:
            seen.append(product)

    ohrc_products = [
        product
        for product in seen
        if (
            str(getattr(product, "mission", "") or "").upper() == "CHANDRAYAAN-2"
            or "CH2" in str(getattr(product, "product_id", "") or "").upper()
        )
        and (
            str(getattr(product, "instrument", "") or "").upper() == "OHRC"
            or "OHRC" in str(getattr(product, "product_id", "") or "").upper()
        )
        and _normalize_status(getattr(product, "status", "")) in {"validated", "partial"}
    ]
    lroc_products = [
        product
        for product in seen
        if str(getattr(product, "mission", "") or "").upper() in {"LRO", "LROC"}
        and str(getattr(product, "instrument", "") or "").upper() in {"NAC", "LROC"}
    ]

    primary_ohrc = sorted(ohrc_products, key=lambda product: (product.gsd_m is not None, float(product.gsd_m or 0.0)))[:1]
    primary_ohrc = primary_ohrc[0] if primary_ohrc else None

    summary = {
        "source": {
            "mission": "Chandrayaan-2",
            "instrument": "OHRC",
            "product_id": getattr(primary_ohrc, "product_id", None),
            "file": str(getattr(primary_ohrc, "image_path", "")),
            "resolution_m_per_px": getattr(primary_ohrc, "gsd_m", None),
            "verified": primary_ohrc is not None,
        },
        "reference": {
            "mission": "LRO",
            "instrument": "LROC NAC",
            "verified": bool(lroc_products),
            "status": "FOUND" if lroc_products else "NOT FOUND",
            "product_id": getattr(lroc_products[0], "product_id", None) if lroc_products else None,
            "file": str(getattr(lroc_products[0], "image_path", "")) if lroc_products else None,
        },
        "pair_available": bool(primary_ohrc is not None and lroc_products),
    }

    print("REAL DATA DISCOVERY")
    print("")
    print("Chandrayaan-2")
    print("  OHRC")
    if primary_ohrc is not None:
        print(f"    Product ID: {primary_ohrc.product_id}")
        print(f"    File: {primary_ohrc.image_path}")
        print(f"    Resolution: {primary_ohrc.gsd_m} m/pixel")
        print("    Status: VERIFIED")
    else:
        print("    Status: NOT FOUND")
    print("")
    print("LROC")
    print("  NAC")
    if lroc_products:
        print(f"    Product ID: {lroc_products[0].product_id}")
        print(f"    File: {lroc_products[0].image_path}")
        print("    Status: VERIFIED")
    else:
        print("    Status: NOT FOUND")
    print("")
    print("Potential pairs:")
    if primary_ohrc is not None and lroc_products:
        print(f"    {primary_ohrc.product_id} ↔ {lroc_products[0].product_id}")
    else:
        print("    No verified real pair available")

    return summary


def validate_real_pair_cli(pair_id: str, root: str | Path, *, json_output: bool = False) -> dict[str, Any]:
    from samanvaya.validation.run_real_pair import validate_real_pair

    manifest_path = Path(root) / "data" / "real" / "manifest.json"
    result = validate_real_pair(pair_id, manifest_path=manifest_path)
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Pair: {pair_id}")
        print(f"Status: {result['status']}")
        print(f"Ground truth: {result['ground_truth_status']}")
        print(f"Reason: {result['reason']}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="samanvaya", description="Samanvaya real-data discovery and mission workflow tools")
    subparsers = parser.add_subparsers(dest="command")

    discover = subparsers.add_parser("discover-data", help="Discover verified mission products in the local repository")
    discover.add_argument("--root", default=".", help="Root directory to scan for candidate planetary products")
    discover.add_argument("--json", action="store_true", help="Emit the discovery summary as JSON")

    validate = subparsers.add_parser("validate-real", help="Conservatively validate a real imported OHRC ↔ LROC pair")
    validate.add_argument("pair_id", help="Pair ID from the real-data manifest")
    validate.add_argument("--root", default=".", help="Repository root containing data/real/manifest.json")
    validate.add_argument("--json", action="store_true", help="Emit validation result as JSON")

    args = parser.parse_args(argv)
    if args.command == "discover-data":
        summary = discover_real_data(args.root)
        if args.json:
            print(json.dumps(summary, indent=2))
        return 0
    if args.command == "validate-real":
        validate_real_pair_cli(args.pair_id, args.root, json_output=args.json)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
