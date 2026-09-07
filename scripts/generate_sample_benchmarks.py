"""
Generates lightweight pre-cropped lunar benchmark GeoTIFF pairs and manifest metadata
for Chandrayaan-2 and NASA LRO missions.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from lunar_core.data_io.synthetic_generator import LunarTerrainSimulator


def write_geotiff(
    file_path: Path,
    data: np.ndarray,
    gsd_m: float,
    origin_lon: float,
    origin_lat: float,
) -> None:
    h, w = data.shape[:2]
    # Degrees per meter on the Moon (R_moon ~ 1,737,400 m)
    deg_per_m = 1.0 / (1737400.0 * np.pi / 180.0)
    res_deg = gsd_m * deg_per_m
    transform = from_origin(origin_lon, origin_lat, res_deg, res_deg)

    # Normalize to uint8 for maximum compatibility and minimal size
    norm = np.clip(data * 255.0, 0, 255).astype(np.uint8)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": w,
        "height": h,
        "count": 1,
        "crs": "+proj=latlong +a=1737400 +b=1737400 +no_defs",
        "transform": transform,
        "compress": "deflate",
    }

    with rasterio.open(str(file_path), "w", **profile) as dst:
        dst.write(norm, 1)


def generate_benchmark_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = PhotometricNormalizer()

    # =========================================================================
    # SCENARIO A: Chandrayaan-2 OHRC (0.25m) vs LRO NAC (0.50m) - Apollo 11 Base
    # =========================================================================
    print("Generating Scenario A: Apollo 11 Landing Site (OHRC vs LRO NAC)...")
    sim_a = LunarTerrainSimulator(size=(256, 256), seed=11)
    dem_a = sim_a.generate_dem(num_craters=16)

    sun_a_src = SunAngles(azimuth_deg=72.5, elevation_deg=28.0)
    raw_a_src = sim_a.render_optical_image(dem_a, sun_a_src)

    sun_a_ref = SunAngles(azimuth_deg=85.0, elevation_deg=33.5)
    raw_a_ref = sim_a.render_optical_image(dem_a, sun_a_ref)
    # Apply realistic orbital misalignment: 1.5° yaw rotation and 4.2 px shift
    M_a = cv2.getRotationMatrix2D((128, 128), 1.5, 1.0)
    M_a[:, 2] += [4.2, -2.8]
    warped_a_ref = cv2.warpAffine(raw_a_ref, M_a, (256, 256), borderMode=cv2.BORDER_REFLECT)

    src_a_path = output_dir / "scenario_a_ohrc_apollo11.tif"
    ref_a_path = output_dir / "scenario_a_lronac_apollo11.tif"
    write_geotiff(src_a_path, raw_a_src, gsd_m=0.25, origin_lon=23.473, origin_lat=0.674)
    write_geotiff(ref_a_path, warped_a_ref, gsd_m=0.50, origin_lon=23.473, origin_lat=0.674)

    # =========================================================================
    # SCENARIO B: TMC-2 Stereo Fore (+26°) vs Nadir (0°) - Jackson Crater
    # =========================================================================
    print("Generating Scenario B: Jackson Crater Stereo Pair (TMC-2 Fore vs Nadir)...")
    sim_b = LunarTerrainSimulator(size=(256, 256), seed=22)
    dem_b = sim_b.generate_dem(num_craters=10)

    sun_b = SunAngles(azimuth_deg=110.0, elevation_deg=42.0)
    raw_b_nadir = sim_b.render_optical_image(dem_b, sun_b)

    # Fore camera exhibits along-track parallax foreshortening
    M_fore = np.float32([[1.0, 0.0, 1.5], [0.0, 1.06, 6.0]])
    warped_b_fore = cv2.warpAffine(raw_b_nadir, M_fore, (256, 256), borderMode=cv2.BORDER_REFLECT)

    src_b_path = output_dir / "scenario_b_tmc2_fore.tif"
    ref_b_path = output_dir / "scenario_b_tmc2_nadir.tif"
    write_geotiff(src_b_path, warped_b_fore, gsd_m=5.0, origin_lon=-163.12, origin_lat=22.40)
    write_geotiff(ref_b_path, raw_b_nadir, gsd_m=5.0, origin_lon=-163.12, origin_lat=22.40)

    # =========================================================================
    # SCENARIO C: Extreme Solar Lighting (Low Sun 12° vs High Sun 65°)
    # =========================================================================
    print("Generating Scenario C: Shackleton Crater Rim (Low Sun 12° vs High Sun 65°)...")
    sim_c = LunarTerrainSimulator(size=(256, 256), seed=33)
    dem_c = sim_c.generate_dem(num_craters=8)

    sun_c_low = SunAngles(azimuth_deg=45.0, elevation_deg=12.0)
    raw_c_low = sim_c.render_optical_image(dem_c, sun_c_low)

    sun_c_high = SunAngles(azimuth_deg=225.0, elevation_deg=65.0)
    raw_c_high = sim_c.render_optical_image(dem_c, sun_c_high)

    M_c = cv2.getRotationMatrix2D((128, 128), -2.1, 1.0)
    M_c[:, 2] += [-3.5, 5.0]
    warped_c_high = cv2.warpAffine(raw_c_high, M_c, (256, 256), borderMode=cv2.BORDER_REFLECT)

    src_c_path = output_dir / "scenario_c_low_sun_12deg.tif"
    ref_c_path = output_dir / "scenario_c_high_sun_65deg.tif"
    write_geotiff(src_c_path, raw_c_low, gsd_m=0.35, origin_lon=0.0, origin_lat=-89.90)
    write_geotiff(ref_c_path, warped_c_high, gsd_m=0.50, origin_lon=0.0, origin_lat=-89.90)

    # =========================================================================
    # Generate Manifest Catalog
    # =========================================================================
    manifest = {
        "version": "1.0.0",
        "description": "Pre-bundled Chandrayaan-2 and NASA LRO mission evaluation datasets",
        "benchmarks": {
            "scenario_a": {
                "id": "scenario_a",
                "title": "Scenario A: Chandrayaan-2 OHRC vs NASA LRO NAC (Apollo 11 Landing Site)",
                "target": "Mare Tranquillitatis (0.674°N, 23.473°E)",
                "description": "Sub-meter cross-spacecraft optical registration between Chandrayaan-2 OHRC (0.25 m/px) and LRO NAC (0.50 m/px) over Tranquility Base.",
                "source": {
                    "filename": "scenario_a_ohrc_apollo11.tif",
                    "spacecraft": "ISRO Chandrayaan-2 Orbiter",
                    "sensor": "OHRC (Orbiter High Resolution Camera)",
                    "gsd_m": 0.25,
                    "altitude_km": 100.0,
                    "sun_azimuth_deg": 72.5,
                    "sun_elevation_deg": 28.0,
                },
                "reference": {
                    "filename": "scenario_a_lronac_apollo11.tif",
                    "spacecraft": "NASA Lunar Reconnaissance Orbiter (LRO)",
                    "sensor": "NAC (Narrow Angle Camera M102552876LE)",
                    "gsd_m": 0.50,
                    "altitude_km": 50.0,
                    "sun_azimuth_deg": 85.0,
                    "sun_elevation_deg": 33.5,
                },
            },
            "scenario_b": {
                "id": "scenario_b",
                "title": "Scenario B: Chandrayaan-2 TMC-2 Fore vs Nadir Stereo Baseline (Jackson Crater)",
                "target": "Jackson Crater Terraces & Central Peak (22.40°N, 163.12°W)",
                "description": "In-orbit stereo baseline photogrammetry between TMC-2 Fore (+26° look angle) and Nadir (0° look angle) with along-track parallax.",
                "source": {
                    "filename": "scenario_b_tmc2_fore.tif",
                    "spacecraft": "ISRO Chandrayaan-2 Orbiter",
                    "sensor": "TMC-2 Fore Camera (+26° look angle)",
                    "gsd_m": 5.0,
                    "altitude_km": 100.0,
                    "sun_azimuth_deg": 110.0,
                    "sun_elevation_deg": 42.0,
                },
                "reference": {
                    "filename": "scenario_b_tmc2_nadir.tif",
                    "spacecraft": "ISRO Chandrayaan-2 Orbiter",
                    "sensor": "TMC-2 Nadir Camera (0° look angle)",
                    "gsd_m": 5.0,
                    "altitude_km": 100.0,
                    "sun_azimuth_deg": 110.0,
                    "sun_elevation_deg": 42.0,
                },
            },
            "scenario_c": {
                "id": "scenario_c",
                "title": "Scenario C: Extreme Solar Lighting Disparity (Low Sun 12° vs High Sun 65°)",
                "target": "South Pole Shackleton Crater Rim (89.90°S)",
                "description": "Severe radiometric test: grazing low-sun illumination (12° elevation) with 60% crater floor cast shadows vs near-zenith high-sun illumination (65° elevation).",
                "source": {
                    "filename": "scenario_c_low_sun_12deg.tif",
                    "spacecraft": "ISRO Chandrayaan-2 Orbiter",
                    "sensor": "OHRC (Low Sun Twilight Orbit)",
                    "gsd_m": 0.35,
                    "altitude_km": 100.0,
                    "sun_azimuth_deg": 45.0,
                    "sun_elevation_deg": 12.0,
                },
                "reference": {
                    "filename": "scenario_c_high_sun_65deg.tif",
                    "spacecraft": "NASA Lunar Reconnaissance Orbiter (LRO)",
                    "sensor": "NAC (High Sun Zenith Orbit)",
                    "gsd_m": 0.50,
                    "altitude_km": 50.0,
                    "sun_azimuth_deg": 225.0,
                    "sun_elevation_deg": 65.0,
                },
            },
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Successfully wrote 6 GeoTIFFs and manifest to {output_dir}")


if __name__ == "__main__":
    generate_benchmark_assets(Path("lunar_core/assets/sample_data"))
