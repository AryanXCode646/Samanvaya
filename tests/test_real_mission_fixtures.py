import json
from pathlib import Path

import numpy as np

from lunar_core.data_io.mission_adapters import adapter_for_mission
from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.data_io.discovery import discover_products, discover_reference_products
from scripts.register_real_pair import register_pair


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real_mission"


def _copy_fixture_to_image(fixture_name: str, image_name: str, tmp_path: Path) -> Path:
    fixture = FIXTURE_DIR / fixture_name
    image_path = tmp_path / image_name
    image_path.write_bytes(b"\x00" * 64)
    image_path.with_suffix(".xml").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    return image_path


def test_chandrayaan2_real_metadata_fixture_identifies_product(tmp_path: Path):
    image_path = _copy_fixture_to_image("ch2_ohr_real_metadata.xml", "CH2_OHR_20250612_0001.img", tmp_path)
    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.gsd_m == 0.28
    assert product.status == "validated"
    assert product.footprint is not None


def test_lro_real_metadata_fixture_identifies_product(tmp_path: Path):
    image_path = _copy_fixture_to_image("lro_nac_real_metadata.xml", "LRO_LROCNAC_0001.img", tmp_path)
    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.mission == "LRO"
    assert product.instrument == "NAC"
    assert product.gsd_m == 0.50
    assert product.status == "validated"
    assert product.footprint is not None


def test_mission_adapter_uses_aliases_and_validation():
    adapter = adapter_for_mission("chandrayaan-2")
    assert adapter is not None
    assert adapter.supported_instruments == ("OHRC", "TMC-2", "IIRS")
    assert adapter.matches_mission("Chandrayaan-2")
    assert adapter.matches_mission("ch2")
    assert adapter.matches_instrument("OHRC")
    assert adapter.matches_instrument("ohr")


def test_register_pair_api_is_available():
    import scripts.register_real_pair as module

    assert hasattr(module, "register_pair")


def test_pds4_label_to_geotiff_product_chain_checks_dimensions(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    image_path = tmp_path / "CH2_OHR_CHAIN.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.zeros((1, 8, 8), dtype=np.uint8))
    image_path.with_suffix(".xml").write_text(
        """<Product_Observational>
        <mission_name>Chandrayaan-2</mission_name><instrument_id>OHRC</instrument_id>
        <product_id>CH2_OHR_CHAIN</product_id><ground_sample_distance>0.28</ground_sample_distance>
        <Array_2D_Image><Axis_Array><elements>8</elements></Axis_Array><Axis_Array><elements>8</elements></Axis_Array></Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = discover_products(tmp_path)[0]

    assert product.status == "validated"
    assert product.image_path == image_path
    assert product.width == 8 and product.height == 8
    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.gsd_m == 0.28


def test_discovery_keeps_reference_mission_classes_separate(tmp_path: Path):
    _copy_fixture_to_image("lro_nac_real_metadata.xml", "LRO_LROCNAC_0001.img", tmp_path)
    _copy_fixture_to_image("ch2_ohr_real_metadata.xml", "CH2_OHR_20250612_0001.img", tmp_path)

    references = discover_reference_products(tmp_path)

    assert references
    assert all(product.mission in {"LRO", "SELENE"} for product in references)


def test_register_pair_accepts_ground_truth_control_points_and_writes_provenance(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_path = raw_dir / "source.tif"
    target_path = raw_dir / "target.tif"
    gt_path = tmp_path / "ground_truth.json"

    source = np.random.default_rng(0).normal(0.5, 0.05, size=(64, 64)).astype(np.float32)
    target = np.random.default_rng(1).normal(0.5, 0.05, size=(64, 64)).astype(np.float32)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=source.shape[0],
        width=source.shape[1],
        count=1,
        dtype=source.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(source, 1)
    with rasterio.open(
        target_path,
        "w",
        driver="GTiff",
        height=target.shape[0],
        width=target.shape[1],
        count=1,
        dtype=target.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(target, 1)

    gt_data = {
        "source_points": [[10.0, 10.0], [45.0, 12.0], [11.0, 48.0], [47.0, 49.0]],
        "target_points": [[8.0, 9.0], [42.0, 11.0], [10.0, 46.0], [43.0, 48.0]],
    }
    gt_path.write_text(json.dumps(gt_data), encoding="utf-8")

    output_dir = tmp_path / "results"
    exit_code = register_pair(
        raw_dir=raw_dir,
        output_dir=output_dir,
        source=str(source_path),
        target=str(target_path),
        ground_truth=str(gt_path),
    )

    assert exit_code == 0
    metrics = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert metrics["metadata"]["ground_truth_available"] is True
    assert provenance["execution"]["ground_truth_available"] is True
    assert provenance["execution"]["control_point_file"] == str(gt_path)
