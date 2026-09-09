from pathlib import Path

import numpy as np

from scripts.register_real_pair import open_pds4_memmap


def test_open_pds4_memmap_does_not_materialize_the_full_image(tmp_path: Path):
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

    mapped = open_pds4_memmap(image, label)

    assert isinstance(mapped, np.memmap)
    assert mapped.shape == (4, 5)
    assert mapped[0, 0] == 0
    assert mapped[-1, -1] == 19