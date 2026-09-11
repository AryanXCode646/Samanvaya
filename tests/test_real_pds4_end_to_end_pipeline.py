"""Real PDS4 End-to-End Fixture Integration Test.

Tests the full unmocked registration pipeline on realistic cratered lunar terrain
paired with authoritative PDS4 XML metadata labels and georeferenced GeoTIFFs,
verifying that the complete evidence bundle and diagnostic artifacts are produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_core.data_io.mission_catalog import inspect_product
from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator
from lunar_core.models import SunAngles
from samanvaya.validation.real_registration import register_products


def test_real_pds4_end_to_end_pipeline(tmp_path: Path):
    source_dir = tmp_path / "source"
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "registration_output"
    source_dir.mkdir()
    ref_dir.mkdir()

    # 1. Synthesize realistic lunar scene with craters and Lambertian illumination
    sim = LunarTerrainSimulator(size=(512, 512), seed=42)
    sun_source = SunAngles(azimuth_deg=195.4, elevation_deg=34.2)
    sun_ref = SunAngles(azimuth_deg=142.1, elevation_deg=48.5)

    ref_img, src_img, true_mat, _ = sim.generate_registered_pair_with_ground_truth(
        sun_ref=sun_ref,
        sun_tgt=sun_source,
        true_translation=(5.0, -4.0),
        true_rotation_deg=0.0,
    )
    ref_img = ref_img.astype(np.float32)
    src_img = src_img.astype(np.float32)

    # 2. Write GeoTIFF files with Moon spheroid georeferencing
    src_tif = source_dir / "CH2_OHR_NCP_20260612T000000000_D_IMG_D18.tif"
    ref_tif = ref_dir / "M1142582844LC.tif"

    moon_crs = "EPSG:4326"
    src_transform = from_origin(20.0, 12.0, 0.50 / 30300.0, 0.50 / 30300.0)
    ref_transform = from_origin(20.0, 12.0, 0.50 / 30300.0, 0.50 / 30300.0)

    for p, img, trans in [(src_tif, src_img, src_transform), (ref_tif, ref_img, ref_transform)]:
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=img.shape[0],
            width=img.shape[1],
            count=1,
            dtype="float32",
            crs=moon_crs,
            transform=trans,
        ) as ds:
            ds.write(img, 1)

    # 3. Write real-spec PDS4 XML labels
    src_xml = src_tif.with_suffix(".xml")
    src_xml.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:ohrc:product:CH2_OHR_NCP_20260612T000000000_D_IMG_D18</logical_identifier>
    <version_id>1.0</version_id>
    <title>Chandrayaan-2 OHRC Calibrated Observation</title>
  </Identification_Area>
  <Observation_Area>
    <Investigation_Area>
      <name>Chandrayaan-2</name>
      <type>Mission</type>
    </Investigation_Area>
    <Observing_System>
      <Observing_System_Component>
        <name>Orbiter High Resolution Camera</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>
    <Target_Identification>
      <name>Moon</name>
      <type>Satellite</type>
    </Target_Identification>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>{src_tif.name}</file_name>
    </File>
    <Array_2D_Image>
      <Axis_Array>
        <axis_name>LINE</axis_name>
        <elements>512</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>SAMPLE</axis_name>
        <elements>512</elements>
      </Axis_Array>
      <Element_Array>
        <data_type>IEEE754MSBSingle</data_type>
      </Element_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <Mission_Area>
    <product_id>CH2_OHR_NCP_20260612T000000000_D_IMG_D18</product_id>
    <instrument_id>OHRC</instrument_id>
    <ground_sample_distance unit="m">0.50</ground_sample_distance>
    <solar_azimuth_angle unit="deg">195.4</solar_azimuth_angle>
    <solar_elevation_angle unit="deg">34.2</solar_elevation_angle>
    <upper_left_latitude unit="deg">12.0</upper_left_latitude>
    <upper_left_longitude unit="deg">20.0</upper_left_longitude>
    <lower_right_latitude unit="deg">10.0</lower_right_latitude>
    <lower_right_longitude unit="deg">22.0</lower_right_longitude>
  </Mission_Area>
</Product_Observational>
""", encoding="utf-8")

    ref_xml = ref_tif.with_suffix(".xml")
    ref_xml.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:nasa:pds:lro_lroc_nac:data:M1142582844LC</logical_identifier>
    <version_id>1.0</version_id>
    <title>LRO LROC NAC Calibrated Radiance</title>
  </Identification_Area>
  <Observation_Area>
    <Investigation_Area>
      <name>LRO</name>
      <type>Mission</type>
    </Investigation_Area>
    <Observing_System>
      <Observing_System_Component>
        <name>Narrow Angle Camera</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>
    <Target_Identification>
      <name>Moon</name>
      <type>Satellite</type>
    </Target_Identification>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>{ref_tif.name}</file_name>
    </File>
    <Array_2D_Image>
      <Axis_Array>
        <axis_name>LINE</axis_name>
        <elements>512</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>SAMPLE</axis_name>
        <elements>512</elements>
      </Axis_Array>
      <Element_Array>
        <data_type>IEEE754MSBSingle</data_type>
      </Element_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <Mission_Area>
    <product_id>M1142582844LC</product_id>
    <instrument_id>NAC</instrument_id>
    <ground_sample_distance unit="m">0.50</ground_sample_distance>
    <solar_azimuth_angle unit="deg">142.1</solar_azimuth_angle>
    <solar_elevation_angle unit="deg">48.5</solar_elevation_angle>
    <upper_left_latitude unit="deg">12.0</upper_left_latitude>
    <upper_left_longitude unit="deg">20.0</upper_left_longitude>
    <lower_right_latitude unit="deg">10.0</lower_right_latitude>
    <lower_right_longitude unit="deg">22.0</lower_right_longitude>
  </Mission_Area>
</Product_Observational>
""", encoding="utf-8")

    # 4. Inspect products via mission catalog
    source_prod = inspect_product(src_tif, root_dir=source_dir)
    ref_prod = inspect_product(ref_tif, root_dir=ref_dir)

    assert source_prod.mission == "Chandrayaan-2"
    assert source_prod.instrument == "OHRC"
    assert source_prod.gsd_m == 0.50
    assert source_prod.sun_azimuth_deg == 195.4
    assert source_prod.sun_elevation_deg == 34.2
    assert str(source_prod.status).lower() in {"validated", "productstatus.validated"}

    assert ref_prod.mission == "LRO"
    assert ref_prod.instrument == "NAC"
    assert ref_prod.gsd_m == 0.50
    assert ref_prod.sun_azimuth_deg == 142.1
    assert ref_prod.sun_elevation_deg == 48.5
    assert str(ref_prod.status).lower() in {"validated", "productstatus.validated"}

    # 5. Run end-to-end registration pipeline (no mocks)
    res = register_products(source_prod, ref_prod, output_dir=out_dir)

    assert res.status == "SUCCESS"
    assert res.inlier_count is not None and res.inlier_count >= 4
    assert res.transform_parameters is not None
    assert res.subpixel_count is not None and res.subpixel_count > 0

    # 6. Verify complete publication evidence bundle
    assert (out_dir / "registered_source.tif").exists()
    assert (out_dir / "registered.tif").exists()
    assert (out_dir / "matches.csv").exists()
    assert (out_dir / "output_validation.json").exists()
    assert (out_dir / "geometry.json").exists()
    assert (out_dir / "source_metadata.json").exists()
    assert (out_dir / "reference_metadata.json").exists()
    assert (out_dir / "overlap.json").exists()
    assert (out_dir / "provenance.json").exists()
    assert (out_dir / "coordinate_audit.json").exists()
    assert (out_dir / "spatial_distribution.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "transform.json").exists()
    assert (out_dir / "diagnostic_dashboard.png").exists()
    assert (out_dir / "overlay.png").exists()
    assert (out_dir / "residual_vectors.png").exists()

    # Verify georeferencing of registered raster
    with rasterio.open(out_dir / "registered_source.tif") as r_ds:
        assert r_ds.width == 512
        assert r_ds.height == 512
        assert r_ds.count == 1
        arr = r_ds.read(1)
        assert np.isfinite(arr).any()

    # Verify spatial distribution entropy and coverage
    spatial_data = json.loads((out_dir / "spatial_distribution.json").read_text(encoding="utf-8"))
    assert "spatial_entropy" in spatial_data
    assert "coverage_fraction" in spatial_data
    assert "cluster_status" in spatial_data

    # Verify reprojection residual stats
    metrics_data = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_data["rmse_pixels"] < 1.0
    assert metrics_data["inlier_ratio"] > 0.0
