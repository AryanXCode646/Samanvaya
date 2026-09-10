"""Tests for metadata-only mission product discovery."""

from pathlib import Path

import numpy as np

from lunar_core.data_io.mission_catalog import inspect_product, scan
from lunar_core.data_io.mission_adapters import Chandrayaan2Adapter
from lunar_core.data_io.mission_product import MissionProduct
from lunar_core.data_io.pair_selector import propose_pair


def test_mission_product_starts_as_discovered():
  assert MissionProduct().status == "discovered"


def test_catalog_inspects_detached_ohrc_product_without_loading_pixels(tmp_path: Path):
    image_path = tmp_path / "ch2_ohr_nrp_example.img"
    label_path = image_path.with_suffix(".xml")
    label_path.write_text(
        """<?xml version=\"1.0\"?>
<Product_Observational xmlns:isda=\"urn:isda\">
  <start_date_time>2025-06-12T20:31:04Z</start_date_time>
  <processing_level>Raw</processing_level>
  <product_class>Product_Observational</product_class>
  <mission_name>Chandrayaan-2</mission_name>
  <instrument_id>OHRC</instrument_id>
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
    assert product.identification_method == "pds4_metadata"
    assert product.label_association_method == "same_stem"
    assert product.width == 5
    assert product.height == 4
    assert product.gsd_m == 0.28
    assert product.gsd_source == "pds4"
    assert product.sun_azimuth_deg == 201.2
    assert product.sun_geometry_source == "pds4"
    assert product.center_lat_deg == 11.0
    assert product.center_lon_deg == 21.0
    assert product.footprint == [(20.0, 10.0), (22.0, 10.0), (22.0, 12.0), (20.0, 12.0)]


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


def test_catalog_classifies_structured_hysi_cube_as_partial(tmp_path: Path):
    image_path = tmp_path / "ch1_hys_product.qub"
    image_path.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <Identification_Area>
            <logical_identifier>urn:isro:isda:ch1_cho.iir:data_calibrated:ch1_hys_product</logical_identifier>
          </Identification_Area>
          <Observation_Area>
            <Investigation_Area><name>Chandrayaan-1</name><type>Mission</type></Investigation_Area>
            <Observing_System>
              <Observing_System_Component type="Instrument">
                <name>Hyper Spectral Imager</name>
              </Observing_System_Component>
            </Observing_System>
          </Observation_Area>
          <Mission_Area><instrument_id>HYSI</instrument_id></Mission_Area>
          <File_Area_Observational>
            <Array_3D_Spectrum>
              <Axis_Array><axis_name>BAND</axis_name><elements>64</elements></Axis_Array>
              <Axis_Array><axis_name>LINE</axis_name><elements>4</elements></Axis_Array>
              <Axis_Array><axis_name>SAMPLE</axis_name><elements>5</elements></Axis_Array>
              <Element_Array><data_type>IEEE754LSBSingle</data_type></Element_Array>
            </Array_3D_Spectrum>
        </File_Area_Observational></Product_Observational>""",
        encoding="utf-8",
    )
    image_path.write_bytes(bytes(64 * 4 * 5 * 4))

    product = inspect_product(image_path, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-1"
    assert product.instrument == "HYSI"
    assert product.band_count == 64
    assert product.status == "partial"
    assert product.validation_message.startswith("Spectral cube metadata parsed")


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
    assert products[0].identification_method == "mission_specific_identifier"


def test_pair_selector_does_not_invent_overlap_without_coordinates(tmp_path: Path):
    source = MissionProduct("Chandrayaan-2", "OHRC", "source", tmp_path / "source.img", gsd_m=0.28)
    target = MissionProduct("LRO", "NAC", "target", tmp_path / "target.img", gsd_m=0.5)

    pair = propose_pair(source, target)

    assert pair.status == "insufficient_metadata"
    assert pair.overlap_ratio is None
    assert pair.gsd_ratio == 0.5 / 0.28

def test_pair_selector_marks_nearby_products_as_proximity_candidates(tmp_path: Path):
    source = MissionProduct(
      "Chandrayaan-2", "OHRC", "source", tmp_path / "source.img",
      gsd_m=0.28, center_lat_deg=10.0, center_lon_deg=20.0,
    )
    target = MissionProduct(
      "LRO", "NAC", "target", tmp_path / "target.img",
      gsd_m=0.5, center_lat_deg=10.2, center_lon_deg=20.1,
    )

    pair = propose_pair(source, target)

    assert pair.status == "proximity_candidate"
    assert pair.overlap_status == "UNKNOWN"
    assert pair.geometry_method == "unknown"
    assert pair.reason.startswith("Center proximity")


def test_pair_selector_uses_footprint_intersection_when_available(tmp_path: Path):
    source = MissionProduct(
      "Chandrayaan-2", "OHRC", "source", tmp_path / "source.img",
      footprint=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    )
    target = MissionProduct(
      "LRO", "NAC", "target", tmp_path / "target.img",
      footprint=[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
    )

    pair = propose_pair(source, target)

    assert pair.status == "confirmed_overlap"
    assert pair.overlap_status == "APPROXIMATE"
    assert pair.geometry_method == "planar_bounding_box_approximation"
    assert pair.overlap_ratio == 0.25
    assert pair.intersection_area == 1.0
    assert pair.source_area == 4.0
    assert pair.target_area == 4.0
    assert pair.selection_method == "footprint_intersection"
    assert pair.pair_id == "source__target"
    assert pair.reason.startswith("Footprint overlap")


def test_pair_selector_rejects_disjoint_footprints(tmp_path: Path):
    source = MissionProduct(
      "Chandrayaan-2", "OHRC", "source", tmp_path / "source.img",
      footprint=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    target = MissionProduct(
      "LRO", "NAC", "target", tmp_path / "target.img",
      footprint=[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
    )

    pair = propose_pair(source, target)

    assert pair.status == "rejected"
    assert pair.overlap_status == "APPROXIMATE"
    assert pair.overlap_ratio == 0.0