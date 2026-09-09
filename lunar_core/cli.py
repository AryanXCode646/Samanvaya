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
    selected = [pair for pair in candidates if pair["status"] == "candidate"]
    print(json.dumps(selected if args.candidates_only else candidates, indent=2))


def cmd_register(args: argparse.Namespace) -> None:
    """Register a generic source/target mission-product pair."""
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
