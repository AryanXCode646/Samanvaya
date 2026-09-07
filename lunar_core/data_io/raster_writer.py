"""
Geospatial Exporter (GeoTIFF, Ground Control Points CSV, GeoJSON Vector Field).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin

from lunar_core.models import GeoRaster, KeypointMatch
from lunar_core.data_io.raster_reader import sanitize_path


class PlanetaryRasterWriter:
    """
    Exports registered lunar imagery and photogrammetric control products.
    """

    @staticmethod
    def write_geotiff(
        output_path: Union[str, Path],
        data: np.ndarray,
        reference_raster: GeoRaster,
        updated_transform: Optional[Affine] = None,
    ) -> None:
        """
        Exports registered raster as a standard Moon IAU 2000 GeoTIFF.
        """
        out_path = sanitize_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = data.shape[:2]

        trans = updated_transform if updated_transform is not None else reference_raster.transform
        if trans is None:
            trans = from_origin(0.0, 0.0, reference_raster.gsd_meters, reference_raster.gsd_meters)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": reference_raster.nodata_val if reference_raster.nodata_val is not None else -9999.0,
            "width": w,
            "height": h,
            "count": 1,
            "crs": reference_raster.crs,
            "transform": trans,
            "compress": "lzw",
        }

        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

    write_georaster = write_geotiff

    @staticmethod
    def export_gcp_csv(
        matches: List[KeypointMatch],
        output_path: Union[str, Path],
        ref_transform: Optional[Affine] = None,
        allowed_dir: Optional[Path] = None,
    ) -> None:
        """
        Exports tie-points as standard Ground Control Points (GCPs) for photogrammetric bundle adjustment.
        """
        out_path = sanitize_path(output_path, allowed_dir=allowed_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "gcp_id,pixel_ref,line_ref,pixel_tgt,line_tgt,geo_x,geo_y,residual_px,confidence,sigma_x,sigma_y,cov_xy,weight\n"
        ]
        for idx, m in enumerate(matches):
            rx, ry = m.ref_xy
            tx, ty = m.target_xy
            geo_x, geo_y = ref_transform * (rx, ry) if ref_transform else (rx, ry)
            res = m.residual_error if m.residual_error is not None else 0.0
            sx = m.sigma_x if m.sigma_x is not None else 0.50
            sy = m.sigma_y if m.sigma_y is not None else 0.50
            cxy = m.cov_xy if m.cov_xy is not None else 0.0
            w = m.weight if m.weight is not None else (m.confidence if m.confidence is not None else 1.0)
            lines.append(
                f"{idx},{rx:.4f},{ry:.4f},{tx:.4f},{ty:.4f},{geo_x:.6f},{geo_y:.6f},{res:.4f},{m.confidence:.4f},"
                f"{sx:.6f},{sy:.6f},{cxy:.6f},{w:.6f}\n"
            )

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def export_vector_field_geojson(
        matches: List[KeypointMatch],
        output_path: Union[str, Path],
        ref_transform: Optional[Affine] = None,
        allowed_dir: Optional[Path] = None,
    ) -> None:
        """
        Exports displacement vectors as standard GeoJSON features for QGIS and ArcGIS.
        """
        out_path = sanitize_path(output_path, allowed_dir=allowed_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for idx, m in enumerate(matches):
            rx, ry = m.ref_xy
            tx, ty = m.target_xy
            p_ref = ref_transform * (rx, ry) if ref_transform else (rx, ry)
            p_tgt = ref_transform * (tx, ty) if ref_transform else (tx, ty)
            dx = tx - rx
            dy = ty - ry
            res = m.residual_error if m.residual_error is not None else 0.0

            feat = {
                "type": "Feature",
                "id": idx,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(p_ref[0]), float(p_ref[1])], [float(p_tgt[0]), float(p_tgt[1])]],
                },
                "properties": {
                    "gcp_id": idx,
                    "dx_pixels": float(dx),
                    "dy_pixels": float(dy),
                    "magnitude": float(np.sqrt(dx**2 + dy**2)),
                    "residual_px": float(res),
                    "confidence": float(m.confidence),
                },
            }
            features.append(feat)

        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)
