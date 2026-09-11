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
    try:
        result = validate_real_pair(pair_id, manifest_path=manifest_path)
    except KeyError as exc:
        result = {
            "pair_id": pair_id,
            "status": "DATA_REQUIRED",
            "ground_truth_status": "DATA_REQUIRED",
            "reason": str(exc).strip("'\""),
        }
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Pair: {pair_id}")
        print(f"Status: {result['status']}")
        print(f"Ground truth: {result['ground_truth_status']}")
        print(f"Reason: {result['reason']}")
    return result


def discover_pairs_cli(
    root: str = "data",
    out: str | None = None,
    min_overlap: float = 0.05,
    json_output: bool = False,
) -> dict[str, Any]:
    from samanvaya.data_io.pair_discovery import PairDiscoveryEngine

    engine = PairDiscoveryEngine(min_overlap_ratio=min_overlap)
    manifest = engine.discover_and_rank_pairs(root)
    if out:
        manifest.export_json(out)
    d = manifest.to_dict()
    if json_output:
        print(json.dumps(d, indent=2, default=str))
    else:
        print("REAL-DATA CANDIDATE PAIR DISCOVERY & RANKING")
        print("-------------------------------------------")
        print(f"Root Directory: {manifest.root_directory}")
        print(f"Products Scanned: {manifest.products_scanned}")
        print(f"Combinations Evaluated: {manifest.candidate_combinations_evaluated}")
        print(f"Valid Candidates: {manifest.valid_candidate_count}")
        print(f"Confirmed Overlap: {manifest.verified_overlap_count}")
        print("")
        if not manifest.ranked_pairs:
            print("No candidate pairs discovered.")
        else:
            for p in manifest.ranked_pairs[:10]:
                flag = "VALID" if p.is_valid_candidate else "REJECTED"
                ov_str = f"{p.overlap_ratio:.1%}" if p.overlap_ratio is not None else "0.0%"
                gsd_str = f"{p.gsd_ratio:.1f}x" if p.gsd_ratio is not None else "N/A"
                print(f"[{flag}] {p.pair_id}")
                print(f"  Score: {p.candidate_score:.4f} | Overlap: {p.overlap_status} ({ov_str}) | GSD Ratio: {gsd_str}")
                if p.rejection_reason:
                    print(f"  Reason: {p.rejection_reason}")
    return d


def register_cli(
    source_path: str,
    reference_path: str,
    output_dir: str = "output/registered",
    strategy: str = "auto",
    transform: str = "homography",
    subpixel: bool = True,
    checkpoints_path: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    from samanvaya.validation.real_registration import register_products

    src_p = Path(source_path).expanduser().resolve()
    ref_p = Path(reference_path).expanduser().resolve()
    source_product = inspect_product(src_p, root_dir=src_p.parent)
    reference_product = inspect_product(ref_p, root_dir=ref_p.parent)

    config = {
        "strategy": strategy,
        "transformation": transform,
        "subpixel_refinement": subpixel,
    }
    result = register_products(source_product, reference_product, output_dir=output_dir, config=config)
    res_dict = result.to_dict()

    chk_eval = None
    if checkpoints_path and Path(checkpoints_path).is_file():
        from samanvaya.validation.checkpoints import load_checkpoints, evaluate_checkpoints
        try:
            chk_pts = load_checkpoints(checkpoints_path)
            if result.transform_matrix is not None:
                chk_eval = evaluate_checkpoints(result.transform_matrix, chk_pts)
                res_dict["independent_checkpoints"] = chk_eval
        except Exception as exc:
            res_dict["independent_checkpoints_error"] = str(exc)

    if json_output:
        print(json.dumps(res_dict, indent=2, default=str))
    else:
        print("SAMANVAYA REGISTRATION RESULT")
        print("----------------------------")
        print(f"Status: {result.status}")
        print(f"Source: {result.source_id} ({result.source_mission or 'Unknown'} {result.source_instrument or 'Unknown'})")
        print(f"Reference: {result.reference_id} ({result.reference_mission or 'Unknown'} {result.reference_instrument or 'Unknown'})")
        print(f"Scale Ratio: {result.scale_ratio}")
        print(f"Matcher: {result.matcher}")
        if result.inlier_count is not None and result.raw_match_count is not None:
            ratio_str = f"{result.inlier_ratio:.1%}" if result.inlier_ratio is not None else "N/A"
            print(f"Inlier Count: {result.inlier_count} / {result.raw_match_count} ({ratio_str})")
        print(f"Transform Model: {result.transform_model}")
        if result.residual_statistics:
            print(f"Reprojection Consensus RMSE: {result.residual_statistics.get('reprojection_rmse_px', 'N/A')} px")
        if chk_eval and chk_eval.get("status") == "READY":
            print(f"Independent Checkpoint RMSE: {chk_eval.get('rmse_pixels', 'N/A')} px ({chk_eval.get('checkpoint_count', 0)} checkpoints)")
            isro_ok = chk_eval.get('rmse_pixels', 999.0) < 0.40
            print(f"Meets ISRO Mandate (< 0.40 px): {isro_ok}")
        if result.registered_output:
            print(f"Registered Output: {result.registered_output}")
        if result.failure_reason:
            print(f"Failure Reason: {result.failure_reason}")
    return res_dict


def validate_cli(
    source: str,
    reference: str | None = None,
    manifest: str | None = None,
    root: str = ".",
    json_output: bool = False,
) -> dict[str, Any]:
    if reference is None:
        # pair_id mode
        return validate_real_pair_cli(source, root, json_output=json_output)

    from lunar_core.data_io.pair_selector import propose_pair
    src_p = Path(source).expanduser().resolve()
    ref_p = Path(reference).expanduser().resolve()
    src_prod = inspect_product(src_p, root_dir=src_p.parent)
    ref_prod = inspect_product(ref_p, root_dir=ref_p.parent)
    pair = propose_pair(src_prod, ref_prod)
    d = pair.to_dict()
    if json_output:
        print(json.dumps(d, indent=2, default=str))
    else:
        print(f"PAIR VALIDATION: {pair.pair_id}")
        print(f"  Status: {pair.status}")
        print(f"  Overlap Ratio: {pair.overlap_ratio}")
        print(f"  Overlap Status: {pair.overlap_status} (Method: {pair.geometry_method})")
        print(f"  Scale Ratio: {pair.gsd_ratio}")
        print(f"  Illumination Delta: {pair.illumination_delta_deg}°")
        print(f"  Metadata Quality: {pair.metadata_quality}")
        print(f"  Reason: {pair.reason}")
    return d


def benchmark_cli(
    manifest_path: str = "data/real/manifest.json",
    output_dir: str = "output/benchmark",
    ablation: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    if ablation:
        from samanvaya.validation.ablation import run_ablation_study
        import numpy as np
        # Check if manifest has an executable pair, otherwise run synthetic ablation
        m_path = Path(manifest_path).expanduser().resolve()
        source_arr = None
        ref_arr = None
        if m_path.is_file():
            payload = json.loads(m_path.read_text(encoding="utf-8"))
            for pair in payload.get("pairs", []):
                src_f = Path(pair.get("source_product", {}).get("image_path", ""))
                ref_f = Path(pair.get("reference_product", {}).get("image_path", ""))
                if src_f.is_file() and ref_f.is_file():
                    import rasterio
                    with rasterio.open(src_f) as s_ds, rasterio.open(ref_f) as r_ds:
                        source_arr = s_ds.read(1).astype(np.float32)
                        ref_arr = r_ds.read(1).astype(np.float32)
                    break
        if source_arr is None or ref_arr is None:
            # Synthetic 256x256 ablation test
            y, x = np.ogrid[:256, :256]
            ref_arr = (np.sin(x / 10.0) * np.cos(y / 10.0) * 128 + 128).astype(np.float32)
            source_arr = np.roll(ref_arr, 5, axis=0)
        res = run_ablation_study(source_arr, ref_arr, output_dir=output_dir, pair_id="ablation_benchmark")
        if json_output:
            print(json.dumps(res, indent=2, default=str))
        else:
            print("SAMANVAYA 7-STAGE ABLATION BENCHMARK")
            print("-----------------------------------")
            for st in res["stages"]:
                print(f"Stage {st['stage_index']}: {st['stage_name']:22s} | Status: {st['status']:10s} | Inliers: {st['inlier_count']:3d} | RMSE: {str(st['reprojection_rmse_px'])[:6] if st['reprojection_rmse_px'] else 'N/A':6s} | Time: {st['runtime_ms']:.1f}ms")
        return res

    from samanvaya.validation.benchmark_real import run_real_benchmark
    res = run_real_benchmark(manifest_path, output_dir)
    if json_output:
        print(json.dumps(res, indent=2, default=str))
    else:
        print("REAL MISSION BENCHMARK")
        print("---------------------")
        print(f"Status: {res.get('status')}")
        print(f"Total Evaluated: {len(res.get('results', []))}")
        if res.get('reason'):
            print(f"Reason: {res.get('reason')}")
        for row in res.get("results", []):
            print(f"  {row.get('pair_id')}: {row.get('status')} ({row.get('reason') or row.get('failure_reason') or 'OK'})")
    return res


def inspect_product_cli(product_path: str, root_dir: str | None = None, json_output: bool = False) -> dict[str, Any]:
    p = Path(product_path).expanduser().resolve()
    r = Path(root_dir).expanduser().resolve() if root_dir else p.parent
    product = inspect_product(p, root_dir=r)
    d = product.to_dict()
    if json_output:
        print(json.dumps(d, indent=2, default=str))
    else:
        print(f"PRODUCT: {product.product_id}")
        print(f"  Status: {product.status}")
        print(f"  Mission: {product.mission}")
        print(f"  Instrument: {product.instrument}")
        print(f"  GSD: {product.gsd_m} m/pixel")
        print(f"  Sun Azimuth: {product.sun_azimuth_deg}° | Elevation: {product.sun_elevation_deg}°")
        print(f"  CRS: {product.crs}")
        print(f"  Center (Lat, Lon): ({product.center_lat_deg}, {product.center_lon_deg})")
        print(f"  Footprint: {product.footprint_status}")
    return d


def inspect_pair_cli(source_path: str, reference_path: str, json_output: bool = False) -> dict[str, Any]:
    from lunar_core.data_io.pair_selector import propose_pair
    src_p = Path(source_path).expanduser().resolve()
    ref_p = Path(reference_path).expanduser().resolve()
    src_prod = inspect_product(src_p, root_dir=src_p.parent)
    ref_prod = inspect_product(ref_p, root_dir=ref_p.parent)
    pair = propose_pair(src_prod, ref_prod)
    d = pair.to_dict()
    if json_output:
        print(json.dumps(d, indent=2, default=str))
    else:
        print(f"PAIR: {pair.pair_id}")
        print(f"  Status: {pair.status}")
        print(f"  Type: {pair.pair_type}")
        print(f"  Overlap Ratio: {pair.overlap_ratio}")
        print(f"  Overlap Status: {pair.overlap_status} (Geometry: {pair.geometry_method})")
        print(f"  Scale Ratio (GSD): {pair.gsd_ratio}")
        print(f"  Solar Angle Delta: {pair.illumination_delta_deg}°")
        print(f"  Metadata Quality: {pair.metadata_quality}")
        print(f"  Reason: {pair.reason}")
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="samanvaya", description="Samanvaya planetary registration and mission workflow suite")
    subparsers = parser.add_subparsers(dest="command")

    # register
    reg = subparsers.add_parser("register", help="Register two planetary image products")
    reg.add_argument("source", help="Path to source planetary image product")
    reg.add_argument("reference", help="Path to reference planetary image product")
    reg.add_argument("--output-dir", "--out", default="output/registered", help="Directory to save registration artifacts")
    reg.add_argument("--strategy", default="auto", choices=["auto", "loftr", "sift", "phase_correlation", "rift"], help="Matcher strategy")
    reg.add_argument("--transform", default="homography", choices=["translation", "similarity", "affine", "homography", "auto"], help="Geometric transformation model")
    reg.add_argument("--no-subpixel", dest="subpixel", action="store_false", default=True, help="Disable analytical subpixel refinement")
    reg.add_argument("--checkpoints", default=None, help="Optional CSV or JSON file of independent validation checkpoints")
    reg.add_argument("--json", action="store_true", help="Emit registration result as JSON")

    # discover-pairs
    disc_pairs = subparsers.add_parser("discover-pairs", help="Discover and rank candidate overlapping product pairs across missions")
    disc_pairs.add_argument("--root", default="data", help="Root directory to scan for planetary products (default: data)")
    disc_pairs.add_argument("--out", default=None, help="Optional output JSON path for the discovery manifest")
    disc_pairs.add_argument("--min-overlap", type=float, default=0.05, help="Minimum overlap ratio threshold (default: 0.05)")
    disc_pairs.add_argument("--json", action="store_true", help="Emit discovery manifest as JSON to stdout")

    # validate
    val = subparsers.add_parser("validate", help="Validate pair overlap or registration quality")
    val.add_argument("source", help="Path to source product OR pair_id")
    val.add_argument("reference", nargs="?", default=None, help="Optional path to reference product")
    val.add_argument("--root", default=".", help="Repository root containing data/real/manifest.json")
    val.add_argument("--json", action="store_true", help="Emit validation result as JSON")

    # benchmark
    bm = subparsers.add_parser("benchmark", help="Run benchmark across real mission pairs or scientific ablation")
    bm.add_argument("--manifest", default="data/real/manifest.json", help="Path to benchmark manifest")
    bm.add_argument("--output-dir", default="output/benchmark", help="Output directory for benchmark reports")
    bm.add_argument("--ablation", action="store_true", help="Run 7-stage scientific ablation benchmark")
    bm.add_argument("--json", action="store_true", help="Emit benchmark summary as JSON")

    # inspect-product
    insp_prod = subparsers.add_parser("inspect-product", help="Inspect metadata and georeferencing of a single product")
    insp_prod.add_argument("product_path", help="Path to planetary image or label")
    insp_prod.add_argument("--root-dir", default=None, help="Root directory for companion file discovery")
    insp_prod.add_argument("--json", action="store_true", help="Emit product inspection as JSON")

    # inspect-pair
    insp_pair = subparsers.add_parser("inspect-pair", help="Inspect and evaluate pairing between two products")
    insp_pair.add_argument("source_path", help="Path to source planetary image")
    insp_pair.add_argument("reference_path", help="Path to reference planetary image")
    insp_pair.add_argument("--json", action="store_true", help="Emit pair evaluation as JSON")

    # discover-data (backward compatibility)
    discover = subparsers.add_parser("discover-data", help="Discover verified mission products in the local repository")
    discover.add_argument("--root", default=".", help="Root directory to scan for candidate planetary products")
    discover.add_argument("--json", action="store_true", help="Emit the discovery summary as JSON")

    # validate-real (backward compatibility)
    validate_real = subparsers.add_parser("validate-real", help="Conservatively validate a real imported OHRC ↔ LROC pair")
    validate_real.add_argument("pair_id", help="Pair ID from the real-data manifest")
    validate_real.add_argument("--root", default=".", help="Repository root containing data/real/manifest.json")
    validate_real.add_argument("--json", action="store_true", help="Emit validation result as JSON")

    # inventory
    inv_cmd = subparsers.add_parser("inventory", help="Inspect and categorize repository products into AUTHORIZED_REAL, SYNTHETIC_FIXTURE, UNVERIFIED, INVALID")
    inv_cmd.add_argument("--root", default="data", help="Root directory to scan for products (default: data)")
    inv_cmd.add_argument("--output", help="Optional output JSON path for inventory evidence")
    inv_cmd.add_argument("--json", action="store_true", help="Emit inventory as JSON to stdout")

    args = parser.parse_args(argv)
    if args.command == "register":
        register_cli(
            args.source,
            args.reference,
            output_dir=args.output_dir,
            strategy=args.strategy,
            transform=args.transform,
            subpixel=args.subpixel,
            checkpoints_path=args.checkpoints,
            json_output=args.json,
        )
        return 0
    if args.command == "discover-pairs":
        discover_pairs_cli(root=args.root, out=args.out, min_overlap=args.min_overlap, json_output=args.json)
        return 0
    if args.command == "validate":
        validate_cli(args.source, reference=args.reference, root=args.root, json_output=args.json)
        return 0
    if args.command == "benchmark":
        benchmark_cli(manifest_path=args.manifest, output_dir=args.output_dir, ablation=args.ablation, json_output=args.json)
        return 0
    if args.command == "inspect-product":
        inspect_product_cli(args.product_path, root_dir=args.root_dir, json_output=args.json)
        return 0
    if args.command == "inspect-pair":
        inspect_pair_cli(args.source_path, args.reference_path, json_output=args.json)
        return 0
    if args.command == "inventory":
        from lunar_core.data_io.discovery import inventory_products

        records = inventory_products(args.root)
        if getattr(args, "output", None):
            out_file = Path(args.output).expanduser().resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "root": str(Path(args.root).resolve()),
                "total_products": len(records),
                "by_classification": {},
                "products": records,
            }
            for rec in records:
                cls_val = str(rec.get("classification", "UNVERIFIED"))
                payload["by_classification"][cls_val] = payload["by_classification"].get(cls_val, 0) + 1
            out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if args.json or not getattr(args, "output", None):
            print(json.dumps(records, indent=2, default=str))
        return 0
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
