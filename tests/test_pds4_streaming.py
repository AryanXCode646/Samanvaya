from pathlib import Path

import numpy as np
from rasterio.windows import Window

from lunar_core.data_io.raster_reader import PlanetaryRasterReader
from lunar_core.data_io.tile_processor import PlanetaryTileProcessor


def _write_sample_product(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "sample.img"
    label = tmp_path / "sample.xml"
    label.write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <file_offset unit="byte">4</file_offset>
          <Array_2D_Image>
            <offset>4</offset>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>5</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )
    image.write_bytes(b"head" + bytes(range(20)))
    return image, label


def test_open_pds4_memmap_does_not_materialize_the_full_image(tmp_path: Path):
    image, label = _write_sample_product(tmp_path)

    mapped = PlanetaryRasterReader.open_pds4_memmap(image, label, allowed_dir=tmp_path)

    assert isinstance(mapped, np.memmap)
    assert mapped.shape == (4, 5)
    assert mapped[0, 0] == 0
    assert mapped[-1, -1] == 19


def test_tile_reader_slices_detached_pds_image(tmp_path: Path):
    image, _label = _write_sample_product(tmp_path)

    tile = PlanetaryTileProcessor._read_window_data(
        image,
        Window(col_off=1, row_off=1, width=2, height=2),
    )

    expected = np.array([[6, 7], [11, 12]], dtype=np.float32)
    p1, p99 = np.percentile(expected, [1.0, 99.0])
    expected = np.clip((expected - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    np.testing.assert_allclose(tile, expected, atol=1e-6)