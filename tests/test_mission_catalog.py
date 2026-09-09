"""Tests for metadata-only mission product discovery."""

from pathlib import Path

import numpy as np

from lunar_core.data_io.mission_catalog import inspect_product, scan
from lunar_core.data_io.mission_adapters import Chandrayaan2Adapter
from lunar_core.data_io.mission_product import MissionProduct
from lunar_core.data_io.pair_selector import propose_pair


def test_catalog_inspects_detached_ohrc_product_without_loading_pixels(tmp_path: Path):
    image_path = tmp_path / "ch2_ohr_nrp_example.img"
    label_path = image_path.with_suffix(".xml")
    label_path.write_text(
        """<?xml version=\"1.0\"?>
<Product_Observational xmlns:isda=\"urn:isda\">
  <start_date_time>2025-06-12T20:31:04Z</start_date_time>
  <processing_level>Raw</processing_level>
  <product_class>Product_Observational</product_class>
  <name>Chandrayaan-2 Orbiter High Resolution Camera OHRC</name>
  <isda:pixel_resolution>0.28</isda:pixel_resolution>
  <isda:sun_azimuth>201.2</isda:sun_azimuth>
  <isda:sun_elevation>28.3</isda:sun_elevation>
  <isda:upper_left_latitude>10.0</isda:upper_left_latitude>
  <isda:lower_right_latitude>12.0</isda:lower_right_latitude>
  <isda:upper_left_longitude>20.0</isda:upper_left_longitude>
  <isda:lower_right_longitude>22.0</isda:lower_right_longitude>
  <Array_2D_Image><offset>0</offset>
    <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
    <Axis_Array><elements>4</elements></Axis_Array>
    <Axis_Array><elements>5</elements></Axis_Array>
  </Array_2D_Image>
</Product_Observational>""",
        encoding="utf-8",
    )
    image_path.write_bytes(bytes(range(20)))

    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.status == "validated"
    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.width == 5
    assert product.height == 4
    assert product.gsd_m == 0.28
    assert product.sun_azimuth_deg == 201.2
    assert product.center_lat_deg == 11.0
    assert product.center_lon_deg == 21.0


def test_catalog_retains_missing_label_failure(tmp_path: Path):
    image_path = tmp_path / "detached.img"
    image_path.write_bytes(np.zeros(8, dtype=np.uint8).tobytes())

    products = scan(tmp_path)

    assert len(products) == 1
    assert products[0].status == "invalid"
    assert "XML label" in products[0].validation_message


def test_catalog_identifies_mission_explicitly_from_metadata(tmp_path: Path):
    image_path = tmp_path / "mission_example.img"
    label_path = image_path.with_suffix(".xml")
    label_path.write_text(
        """<?xml version=\"1.0\"?>
        <Mission_Product>
          <mission_name>Chandrayaan-2</mission_name>
          <instrument_id>OHRC</instrument_id>
          <Array_2D_Image>
            <offset>0</offset>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>5</elements></Axis_Array>
          </Array_2D_Image>
        </Mission_Product>""",
        encoding="utf-8",
    )
    image_path.write_bytes(bytes(range(20)))

    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.status == "validated"


def test_catalog_uses_structured_pds4_product_id_when_available(tmp_path: Path):
    image_path = tmp_path / "random_filename.img"
    label_path = image_path.with_suffix(".xml")
    label_path.write_text(
        """<?xml version=\"1.0\"?>
        <Product_Observational xmlns:img=\"http://pds.nasa.gov\">
          <mission_name>Chandrayaan-2</mission_name>
          <instrument_name>OHRC</instrument_name>
          <product_id>CH2_OHR_1234</product_id>
          <Array_2D_Image>
            <offset>0</offset>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>8</elements></Axis_Array>
            <Axis_Array><elements>6</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )
    image_path.write_bytes(bytes(range(48)))

    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.product_id == "CH2_OHR_1234"
    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.identification_method == "pds4_metadata"


def test_chandrayaan_adapter_recovers_identity_from_product_identifier(tmp_path: Path):
    image_path = tmp_path / "unlabelled_identity.img"
    image_path.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <product_id>CH2_OHR_5678</product_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )
    image_path.write_bytes(bytes(range(16)))

    products = Chandrayaan2Adapter().scan(tmp_path)

    assert len(products) == 1
    assert products[0].mission == "Chandrayaan-2"
    assert products[0].instrument == "OHRC"
    assert products[0].identification_method == "product_id"


def test_pair_selector_does_not_invent_overlap_without_coordinates(tmp_path: Path):
    source = MissionProduct("Chandrayaan-2", "OHRC", "source", tmp_path / "source.img", gsd_m=0.28)
    target = MissionProduct("LRO", "NAC", "target", tmp_path / "target.img", gsd_m=0.5)

    pair = propose_pair(source, target)

    assert pair.status == "insufficient_metadata"
    assert pair.overlap_ratio is None
    assert pair.gsd_ratio == 0.5 / 0.28

def test_pair_selector_marks_nearby_products_as_candidates(tmp_path: Path):
    source = MissionProduct(
      "Chandrayaan-2", "OHRC", "source", tmp_path / "source.img",
      gsd_m=0.28, center_lat_deg=10.0, center_lon_deg=20.0,
    )
    target = MissionProduct(
      "LRO", "NAC", "target", tmp_path / "target.img",
      gsd_m=0.5, center_lat_deg=10.2, center_lon_deg=20.1,
    )

    pair = propose_pair(source, target)

    assert pair.status == "candidate"
    assert pair.reason.startswith("Center distance")