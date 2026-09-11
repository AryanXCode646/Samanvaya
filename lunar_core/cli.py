"""
Samanvaya (समान्वय) Unified Command-Line Interface.
ISRO Chandrayaan-2 Lunar Optical Image Registration Framework (SIH PS 26166).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def print_banner() -> None:
    banner = r"""
  ███████╗ █████╗ ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗ █████╗ ██╗   ██╗ █████╗ 
  ██╔════╝██╔══██╗████╗ ████║██╔══██╗████╗  ██║██║   ██║██╔══██╗╚██╗ ██╔╝██╔══██╗
  ███████╗███████║██╔████╔██║███████║██╔██╗ ██║██║   ██║███████║ ╚████╔╝ ███████║
  ╚════██║██╔══██║██║╚██╔╝██║██╔══██║██║╚██╗██║╚██╗ ██╔╝██╔══██║  ╚██╔╝  ██╔══██║
  ███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║ ╚████╔╝ ██║  ██║   ██║   ██║  ██║
  ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
  ISRO Chandrayaan-2 Planetary Image Registration Framework (SIH PS 26166)
    """
    print(banner)


def cmd_ui(args: argparse.Namespace) -> None:
    """Launches the interactive Streamlit portal."""
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    port = args.port or 8501
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    print(f"🚀 Launching Samanvaya Streamlit Portal on port {port}...")
    subprocess.run(cmd)


def cmd_test(args: argparse.Namespace) -> None:
    """Executes the full automated verification test suite."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    print("🧪 Executing Samanvaya Verification Test Suite...")
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def cmd_align(args: argparse.Namespace) -> None:
    """Headless CLI alignment between Source and Reference lunar GeoTIFFs."""
    from lunar_core.data_io import PlanetaryRasterReader, PlanetaryRasterWriter
    from lunar_core.data_io.raster_reader import sanitize_path
    from lunar_core.alignment.dense_matcher import DenseLoFTRMatcher
    from lunar_core.evaluation.metrics import EvaluationEngine

    src_path = sanitize_path(args.source)
    ref_path = sanitize_path(args.reference)
    out_dir = sanitize_path(args.output or "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading Source Raster: {src_path}")
    raster_src = PlanetaryRasterReader.read_geotiff(src_path)
    print(f"📥 Loading Reference Raster: {ref_path}")
    raster_ref = PlanetaryRasterReader.read_geotiff(ref_path)

    print("⚙️ Executing Dense LoFTR Matching & 2D Parabolic Taylor Sub-pixel Refinement...")
    matcher = DenseLoFTRMatcher(
        pretrained=args.weights,
        confidence_threshold=args.threshold,
        grid_bins=8,
        cap_per_cell=args.cap,
        magsac_reproj_threshold=args.reproj_threshold,
    )
    if not matcher.is_pretrained:
        print("WARNING: LoFTR pretrained weights unavailable; results are not meaningful.")

    match_result = matcher.match(
        source_image=raster_src.data,
        reference_image=raster_ref.data,
    )
    inliers = match_result.inliers
    H = match_result.homography
    warped = match_result.warped_source

    total_matches = len(match_result.all_matches)
    print(f"🎯 Discovered {total_matches} raw matches and {len(inliers)} verified inliers.")

    # Generate Report
    report = EvaluationEngine.generate_report(
        total_matches=total_matches,
        inliers=inliers,
        image_shape=raster_ref.data.shape,
        homography=H,
    )

    json_file = out_dir / "samanvaya_evaluation_report.json"
    report.export_json(json_file)
    print(f"📄 Exported Structured Report: {json_file}")

    plot_file = out_dir / "samanvaya_residual_scatter.png"
    report.export_residual_scatter_plot(plot_file, background_image=raster_ref.data)
    print(f"📈 Exported Residual Scatter Plot: {plot_file}")

    if warped is not None:
        warped_file = out_dir / "registered_source.tif"
        PlanetaryRasterWriter.write_geotiff(output_path=warped_file, data=warped, reference_raster=raster_ref)
        print(f"🗺️ Exported Registered GeoTIFF: {warped_file}")

    # Export USGS ISIS3 Jigsaw GCP CSV
    try:
        from lunar_core.data_io.isis_exporter import IsisGcpExporter
        isis_file = out_dir / "samanvaya_isis3_jigsaw_gcp.csv"
        exporter = IsisGcpExporter()
        exporter.export_pairwise_csv(matches=inliers, ref_raster=raster_ref, output_path=isis_file)
        print(f"🌐 Exported USGS ISIS3 Jigsaw GCPs: {isis_file}")
    except Exception as exc:
        print(f"⚠️ ISIS3 GCP Export skipped: {exc}")

    # Export Executive PDF Mission Report
    try:
        from lunar_core.evaluation.pdf_reporter import MissionReportGenerator
        pdf_file = out_dir / "samanvaya_mission_report.pdf"
        reporter = MissionReportGenerator()
        reporter.generate_report(
            metrics=report,
            matches=inliers,
            output_pdf_path=pdf_file,
            ref_modality=raster_ref.modality if hasattr(raster_ref, "modality") else None,
            target_modality=raster_src.modality if hasattr(raster_src, "modality") else None,
            ref_gsd=raster_ref.gsd_meters if hasattr(raster_ref, "gsd_meters") else 0.5,
            target_gsd=raster_src.gsd_meters if hasattr(raster_src, "gsd_meters") else 0.25,
            ref_sun=raster_ref.sun_angles if hasattr(raster_ref, "sun_angles") else None,
            target_sun=raster_src.sun_angles if hasattr(raster_src, "sun_angles") else None,
        )
        print(f"📑 Exported Executive Mission PDF Report: {pdf_file}")
    except Exception as exc:
        print(f"⚠️ PDF Report generation skipped: {exc}")

    print("\n" + "=" * 50)
    print("📊 SAMANVAYA MISSION KPI SUMMARY")
    print("=" * 50)
    assessment = "NOT ASSESSED (reprojection consensus; no ground truth)"
    if report.ground_truth_available:
        assessment = "PASSED ✅" if report.meets_isro_mandate else "NEEDS REVIEW ⚠️"
    print(f"  Reprojection RMSE: {report.rmse_pixels:.4f} px (ground-truth mandate: {assessment})")
    print(f"  Inlier Count   : {report.inlier_count} verified tie-points")
    print(f"  Inlier Ratio   : {report.inlier_ratio_percent:.2f}%")
    print(f"  Spatial Entropy: {report.spatial_uniformity_entropy:.4f} / 1.0 (Non-clumping score)")
    print("=" * 50)


def cmd_info(args: argparse.Namespace) -> None:
    """Displays hardware, library, and mission configuration details."""
    import torch
    import kornia
    import cv2
    import rasterio

    print_banner()
    print("Mission Configuration:")
    print("  Problem Statement: SIH PS 26166")
    print("  Space Agency     : Indian Space Research Organisation (ISRO)")
    print("  Target Mission   : Chandrayaan-2 (OHRC, TMC-2, IIRS) & NASA LRO NAC")
    print("  Coordinate CRS   : Moon IAU 2000 (IAU2000:30100)")
    print("\nEnvironment & Hardware:")
    print(f"  Python Version  : {sys.version.split()[0]}")
    print(f"  PyTorch Version : {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    print(f"  Kornia Version  : {kornia.__version__}")
    print(f"  OpenCV Version  : {cv2.__version__}")
    print(f"  Rasterio Version: {rasterio.__version__}")
    print("\nOperational Capabilities:")
    print("  • Vectorized Log-Gabor Phase Congruency (Zero-DC Illumination Invariance)")
    print("  • O(1) Parabolic Taylor Sub-pixel Refinement (Strict Negative-Definite Hessian)")
    print("  • Out-of-Core Windowed Tiling for Gigapixel GeoTIFFs")
    print("  • Mission metadata views for OHRC, TMC-2, IIRS, LRO NAC, and SELENE TC")
    print("  • USGS ISIS3 Jigsaw GCP Exporter with Curvature Covariance")
    print("  • Classical RIFT Phase-Congruency Matcher & LoFTR Dense Matcher")
    print("  • Interactive Streamlit application (streamlit run app.py)")


def cmd_catalog_scan(args: argparse.Namespace) -> None:
    """Scan mission products without loading their pixel arrays."""
    from lunar_core.data_io.mission_catalog import scan, write_csv

    products = scan(Path(args.root))
    output = Path(args.output)
    write_csv(products, output)
    invalid = [product for product in products if product.status != "validated"]
    print(f"Cataloged {len(products)} products to {output}")
    if invalid:
        print(f"Invalid products: {len(invalid)}")
        for product in invalid:
            print(f"  - {product.image_path}: {product.validation_message}")
        raise SystemExit(1)


def cmd_catalog_show(args: argparse.Namespace) -> None:
    """Display one catalog manifest as JSON."""
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Catalog manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        print(json.dumps(list(csv.DictReader(stream)), indent=2))


def cmd_inspect_product(args: argparse.Namespace) -> None:
    """Inspect one supplied product without loading its raster pixels."""
    from lunar_core.data_io.mission_catalog import inspect_product

    product = inspect_product(Path(args.product), root_dir=Path(args.root).resolve() if args.root else None)
    print(json.dumps(product.to_dict(), indent=2, default=str))
    if args.require_valid and str(product.status) != "validated":
        raise SystemExit(2)


def cmd_discover_pairs(args: argparse.Namespace) -> None:
    from lunar_core.data_io.discovery import discover_benchmark_pairs

    pairs = discover_benchmark_pairs(args.root, source_mission=args.source_mission)
    print(json.dumps([pair.to_dict() for pair in pairs], indent=2, default=str))


def cmd_inventory(args: argparse.Namespace) -> None:
    from lunar_core.data_io.discovery import inventory_products

    records = inventory_products(args.root)
    if getattr(args, "output", None):
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"root": str(Path(args.root).resolve()), "products": records}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(records, indent=2, default=str))


def cmd_discover_real_pairs(args: argparse.Namespace) -> None:
    from lunar_core.data_io.discovery import discover_benchmark_pairs

    pairs = [pair.to_dict() for pair in discover_benchmark_pairs(args.root)]
    if getattr(args, "output", None):
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"root": str(Path(args.root).resolve()), "pairs": pairs}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(pairs, indent=2, default=str))


def cmd_benchmark_real(args: argparse.Namespace) -> None:
    from samanvaya.validation.benchmark_real import run_real_benchmark

    result = run_real_benchmark(args.manifest, args.output)
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") == "DATA_REQUIRED":
        raise SystemExit(2)


def cmd_verify_output(args: argparse.Namespace) -> None:
    import rasterio

    path = Path(args.output).expanduser().resolve()
    if not path.is_file():
        print(json.dumps({"status": "OUTPUT_VERIFICATION_FAILED", "reason": f"Missing output: {path}"}))
        raise SystemExit(2)
    try:
        with rasterio.open(path) as dataset:
            data = dataset.read(1, masked=True)
            valid = int(data.count())
            result = {
                "status": "VERIFIED" if valid else "OUTPUT_VERIFICATION_FAILED",
                "path": str(path),
                "width": dataset.width,
                "height": dataset.height,
                "dtype": dataset.dtypes[0],
                "crs": str(dataset.crs) if dataset.crs else None,
                "nodata": dataset.nodata,
                "valid_pixels": valid,
            }
    except (OSError, ValueError) as exc:
        result = {"status": "OUTPUT_VERIFICATION_FAILED", "path": str(path), "reason": str(exc)}
    print(json.dumps(result, indent=2))
    if result["status"] != "VERIFIED":
        raise SystemExit(2)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Catalog manifest does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cmd_pair_discover(args: argparse.Namespace) -> None:
    """Report conservative candidate pairs from a catalog manifest."""
    from lunar_core.data_io.mission_product import MissionProduct
    from lunar_core.data_io.pair_selector import propose_pair

    rows = _read_manifest(Path(args.manifest))
    products = []
    for row in rows:
        if row.get("status") != "validated":
            continue
        products.append(
            MissionProduct(
                mission=row.get("mission") or None,
                instrument=row.get("instrument") or None,
                product_id=row["product_id"],
                image_path=Path(row["image_path"]),
                gsd_m=float(row["gsd_m"]) if row.get("gsd_m") else None,
                center_lat_deg=float(row["center_lat_deg"]) if row.get("center_lat_deg") else None,
                center_lon_deg=float(row["center_lon_deg"]) if row.get("center_lon_deg") else None,
                status=row["status"],
            )
        )
    if args.mission:
        products = [product for product in products if product.mission == args.mission]
    candidates = [
        propose_pair(source, target).to_dict()
        for index, source in enumerate(products)
        for target in products[index + 1 :]
        if source.product_id != target.product_id
    ]
    selected = [
        pair
        for pair in candidates
        if pair["status"] in {"confirmed_overlap", "proximity_candidate"}
    ]
    print(json.dumps(selected if args.candidates_only else candidates, indent=2))


def cmd_register(args: argparse.Namespace) -> None:
    """Register a mission-product pair through the importable API when real files exist.

    Preserve legacy CLI compatibility for placeholder inputs used by tests and generic
    workflow wrappers by falling back to the subprocess delegation path when the actual
    source/target files are not available.
    """
    from scripts.register_real_pair import register_pair

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    source_exists = bool(args.source) and Path(args.source).expanduser().exists()
    target_exists = bool(args.target) and Path(args.target).expanduser().exists()
    if source_exists or target_exists or raw_dir.exists():
        try:
            exit_code = register_pair(
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                source=args.source,
                target=args.target,
                site=args.site,
            )
        except Exception as exc:  # pragma: no cover - CLI failure path
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(exit_code)

    runner = _PROJECT_ROOT / "scripts" / "register_real_pair.py"
    command = [
        sys.executable,
        str(runner),
        "--raw-dir",
        args.raw_dir,
        "--output-dir",
        args.output_dir,
        "--source",
        args.source,
        "--target",
        args.target,
        "--site",
        args.site,
    ]
    result = subprocess.run(command)
    raise SystemExit(result.returncode)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Display a previously generated evaluation report without recomputing it."""
    report = Path(args.report)
    if not report.is_file():
        raise FileNotFoundError(f"Evaluation report does not exist: {report}")
    print(report.read_text(encoding="utf-8"))


def cmd_validation_summary(args: argparse.Namespace) -> None:
    """Print a conservative validation summary derived from the evidence manifest."""
    from lunar_core.validation import summarize_validation_evidence

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"Validation manifest does not exist: {manifest}")

    summary = summarize_validation_evidence(manifest)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Scientific validation summary")
    print(f"Validation scope: {summary['validation_scope']}")
    print(f"Real image validation status: {summary['real_image_validation_status']}")
    print(f"Merge-readiness scope: {summary['merge_readiness_scope']}")
    for pair, row in summary["validation_matrix"].items():
        print(f"- {pair}: ingestion={row['ingestion']}; registration={row['registration']}; gt={row['ground_truth']}; status={row['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="samanvaya",
        description="Samanvaya: Industry-Grade Lunar Optical Image Registration Framework (SIH PS 26166)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # samanvaya ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive Streamlit registration portal")
    p_ui.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server")
    p_ui.set_defaults(func=cmd_ui)



    # samanvaya test
    p_test = subparsers.add_parser("test", help="Run automated verification test suite")
    p_test.set_defaults(func=cmd_test)

    # samanvaya align
    p_align = subparsers.add_parser("align", help="Headless CLI alignment between GeoTIFFs")
    p_align.add_argument("--source", "-s", required=True, help="Path to Source GeoTIFF (OHRC/TMC-2)")
    p_align.add_argument("--reference", "-r", required=True, help="Path to Reference GeoTIFF (LRO NAC)")
    p_align.add_argument("--output", "-o", default="output", help="Directory to save aligned products")
    p_align.add_argument("--weights", default="outdoor", help="LoFTR model weights checkpoint")
    p_align.add_argument("--threshold", type=float, default=0.15, help="LoFTR confidence threshold")
    p_align.add_argument("--cap", type=int, default=4, help="ANMS equal cap per 8x8 cell")
    p_align.add_argument("--reproj-threshold", type=float, default=1.5, help="USAC-MAGSAC reprojection threshold")
    p_align.set_defaults(func=cmd_align)

    # samanvaya catalog scan
    p_catalog = subparsers.add_parser("catalog", help="Discover mission products and metadata")
    catalog_commands = p_catalog.add_subparsers(dest="catalog_command", required=True)
    p_catalog_scan = catalog_commands.add_parser("scan", help="Scan a raw mission-data directory")
    p_catalog_scan.add_argument("root", help="Directory containing downloaded mission products")
    p_catalog_scan.add_argument("--output", default="data/metadata/products.csv", help="CSV manifest output path")
    p_catalog_scan.set_defaults(func=cmd_catalog_scan)
    p_catalog_show = catalog_commands.add_parser("show", help="Display a CSV catalog manifest")
    p_catalog_show.add_argument("manifest", help="CSV manifest path")
    p_catalog_show.set_defaults(func=cmd_catalog_show)

    # samanvaya inspect-product
    p_inspect = subparsers.add_parser("inspect-product", help="Inspect one supplied mission product")
    p_inspect.add_argument("product", help="Raster product path")
    p_inspect.add_argument("--root", help="Allowed root directory for path safety")
    p_inspect.add_argument("--require-valid", action="store_true", help="Exit 2 unless the product is validated")
    p_inspect.set_defaults(func=cmd_inspect_product)

    p_discover_pairs = subparsers.add_parser("discover-pairs", help="Discover conservative mission/reference candidates")
    p_discover_pairs.add_argument("root", help="Directory containing supplied products")
    p_discover_pairs.add_argument("--source-mission", help="Restrict source mission")
    p_discover_pairs.set_defaults(func=cmd_discover_pairs)

    p_inventory = subparsers.add_parser("inventory", help="Inventory supplied mission/reference products")
    p_inventory.add_argument("--root", required=True)
    p_inventory.add_argument("--output", default=None, help="Optional output JSON path")
    p_inventory.set_defaults(func=cmd_inventory)

    p_discover_real_pairs = subparsers.add_parser("discover-real-pairs", help="Discover real-data candidate pairs")
    p_discover_real_pairs.add_argument("--root", required=True)
    p_discover_real_pairs.add_argument("--output", default=None, help="Optional output JSON path")
    p_discover_real_pairs.set_defaults(func=cmd_discover_real_pairs)

    p_benchmark_real = subparsers.add_parser("benchmark-real", help="Execute a supplied real-data benchmark manifest")
    p_benchmark_real.add_argument("--manifest", required=True)
    p_benchmark_real.add_argument("--output", required=True)
    p_benchmark_real.set_defaults(func=cmd_benchmark_real)

    p_verify_output = subparsers.add_parser("verify-output", help="Reopen and verify a registered raster")
    p_verify_output.add_argument("output")
    p_verify_output.set_defaults(func=cmd_verify_output)

    # samanvaya pair discover
    p_pair = subparsers.add_parser("pair", help="Discover metadata-supported product pairs")
    pair_commands = p_pair.add_subparsers(dest="pair_command", required=True)
    p_pair_discover = pair_commands.add_parser("discover", help="Discover candidate pairs from a catalog")
    p_pair_discover.add_argument("--manifest", default="data/metadata/products.csv")
    p_pair_discover.add_argument("--mission", help="Restrict candidates to one mission")
    p_pair_discover.add_argument("--candidates-only", action="store_true")
    p_pair_discover.set_defaults(func=cmd_pair_discover)

    # samanvaya register
    p_register = subparsers.add_parser("register", help="Register a supplied mission-product pair")
    p_register.add_argument("--source", required=True, help="Source mission product filename or path")
    p_register.add_argument("--target", required=True, help="Target mission product filename or path")
    p_register.add_argument("--raw-dir", default="data/real/raw")
    p_register.add_argument("--output-dir", default="data/real/results")
    p_register.add_argument("--site", default="unspecified")
    p_register.set_defaults(func=cmd_register)

    # samanvaya evaluate
    p_evaluate = subparsers.add_parser("evaluate", help="Display a generated evaluation report")
    p_evaluate.add_argument("--report", required=True, help="Path to evaluation_report.json")
    p_evaluate.set_defaults(func=cmd_evaluate)

    # samanvaya validation
    p_validation = subparsers.add_parser("validation", help="Display the evidence-backed scientific validation summary")
    p_validation.add_argument("--manifest", default="evidence/real_data_manifest.json", help="Path to the validation manifest")
    p_validation.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable summary")
    p_validation.set_defaults(func=cmd_validation_summary)

    # samanvaya validate-real
    p_validate_real = subparsers.add_parser("validate-real", help="Validate a real imported OHRC ↔ LROC pair with conservative evidence rules")
    p_validate_real.add_argument("--pair-id", required=True, help="Pair ID from the manifest")
    p_validate_real.add_argument("--manifest", default="data/real/manifest.json", help="Real-pair manifest path")
    p_validate_real.add_argument("--json", action="store_true", help="Emit JSON output")
    p_validate_real.set_defaults(func=lambda args: print(json.dumps(__import__("samanvaya.validation.run_real_pair", fromlist=["validate_real_pair"]).validate_real_pair(args.pair_id, manifest_path=args.manifest), indent=2) if args.json else "\n".join([
        f"Pair: {args.pair_id}",
        f"Status: {__import__('samanvaya.validation.run_real_pair', fromlist=['validate_real_pair']).validate_real_pair(args.pair_id, manifest_path=args.manifest)['status']}",
        f"Ground truth: {__import__('samanvaya.validation.run_real_pair', fromlist=['validate_real_pair']).validate_real_pair(args.pair_id, manifest_path=args.manifest)['ground_truth_status']}",
        f"Reason: {__import__('samanvaya.validation.run_real_pair', fromlist=['validate_real_pair']).validate_real_pair(args.pair_id, manifest_path=args.manifest)['reason']}",
    ])))

    # samanvaya info
    p_info = subparsers.add_parser("info", help="Display system and mission configuration")
    p_info.set_defaults(func=cmd_info)

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)

    parsed_args = parser.parse_args()
    if hasattr(parsed_args, "func"):
        parsed_args.func(parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
