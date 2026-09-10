"""Deterministic PDS4 label association and constrained instrument identification."""

from pathlib import Path

from lunar_core.data_io.mission_catalog import inspect_product, resolve_product_label
from lunar_core.data_io.product_identity import LabelAssociationStatus


def _write_image(path: Path, size: int = 16) -> Path:
    path.write_bytes(bytes(range(size)))
    return path


def test_same_stem_label_is_authoritative(tmp_path: Path):
    image = _write_image(tmp_path / "ch2_ohr_scene.img")
    (tmp_path / "unrelated.xml").write_text(
        """<?xml version="1.0"?><Product><file_name>other.img</file_name></Product>""",
        encoding="utf-8",
    )
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <mission_name>Chandrayaan-2</mission_name>
          <instrument_id>OHRC</instrument_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.status is LabelAssociationStatus.RESOLVED
    assert resolution.method.value == "same_stem"
    assert resolution.path == image.with_suffix(".xml")
    assert product.label_path == image.with_suffix(".xml")
    assert product.instrument == "OHRC"
    assert product.identification_method == "pds4_metadata"


def test_explicit_file_name_reference_selects_non_stem_label(tmp_path: Path):
    image = _write_image(tmp_path / "scene.img")
    label = tmp_path / "authoritative_product.xml"
    label.write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <File><file_name>scene.img</file_name></File>
          <mission_name>Chandrayaan-2</mission_name>
          <instrument_id>OHRC</instrument_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )
    (tmp_path / "other.xml").write_text(
        """<?xml version="1.0"?><Product><file_name>different.img</file_name></Product>""",
        encoding="utf-8",
    )

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.method.value == "explicit_file_reference"
    assert resolution.path == label
    assert product.status == "validated"
    assert product.instrument == "OHRC"


def test_lone_unrelated_xml_is_never_selected(tmp_path: Path):
    image = _write_image(tmp_path / "photo.img")
    (tmp_path / "catalog.xml").write_text(
        """<?xml version="1.0"?><Product><title>Collection catalog</title></Product>""",
        encoding="utf-8",
    )

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.status is LabelAssociationStatus.MISSING
    assert resolution.path is None
    assert "never selected" in (resolution.message or "").lower() or "authoritative" in (resolution.message or "").lower()
    assert product.status == "invalid"
    assert product.label_path is None


def test_multiple_xml_files_without_authority_do_not_guess(tmp_path: Path):
    image = _write_image(tmp_path / "photo.img")
    (tmp_path / "a.xml").write_text("""<?xml version="1.0"?><A/>""", encoding="utf-8")
    (tmp_path / "b.xml").write_text("""<?xml version="1.0"?><B/>""", encoding="utf-8")

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.path is None
    assert product.label_path is None
    assert product.status == "invalid"


def test_ambiguous_explicit_references_fail_clearly(tmp_path: Path):
    image = _write_image(tmp_path / "scene.img")
    (tmp_path / "first.xml").write_text(
        """<?xml version="1.0"?><Product><file_name>scene.img</file_name></Product>""",
        encoding="utf-8",
    )
    (tmp_path / "second.xml").write_text(
        """<?xml version="1.0"?><Product><file_name>scene.img</file_name></Product>""",
        encoding="utf-8",
    )

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.status is LabelAssociationStatus.AMBIGUOUS
    assert resolution.path is None
    assert "Refusing to guess" in (resolution.message or "")
    assert product.status == "invalid"
    assert product.label_path is None


def test_incorrect_same_stem_file_name_is_rejected(tmp_path: Path):
    image = _write_image(tmp_path / "scene.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?><Product><file_name>different.img</file_name></Product>""",
        encoding="utf-8",
    )

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.path is None
    assert product.status == "invalid"
    assert "does not reference" in (resolution.message or "")


def test_missing_label_is_invalid_for_detached_image(tmp_path: Path):
    image = _write_image(tmp_path / "detached.img")

    resolution = resolve_product_label(image)
    product = inspect_product(image, root_dir=tmp_path)

    assert resolution.status is LabelAssociationStatus.MISSING
    assert product.status == "invalid"
    assert "XML label" in (product.validation_message or "")


def test_structured_instrument_id_is_preferred(tmp_path: Path):
    image = _write_image(tmp_path / "random_filename.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <mission_name>Chandrayaan-2</mission_name>
          <instrument_id>OHRC</instrument_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = inspect_product(image, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.identification_method == "pds4_metadata"


def test_mission_specific_product_id_recovers_ohrc_without_generic_substring(tmp_path: Path):
    image = _write_image(tmp_path / "unlabelled_identity.img")
    image.with_suffix(".xml").write_text(
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

    product = inspect_product(image, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.identification_method == "mission_specific_identifier"


def test_filename_heuristic_is_explicitly_labeled(tmp_path: Path):
    image = _write_image(tmp_path / "ch2_ohr_orbit.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = inspect_product(image, root_dir=tmp_path)

    assert product.mission == "Chandrayaan-2"
    assert product.instrument == "OHRC"
    assert product.identification_method == "filename_heuristic"


def test_generic_tc_text_is_not_selene_terrain_camera(tmp_path: Path):
    image = _write_image(tmp_path / "arbitrary.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <title>Collection notes</title>
          <comment>Terrain Classification (TC) discussion for this archive.</comment>
          <description>STATIC CONTACT PRODUCT CATALOG</description>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = inspect_product(image, root_dir=tmp_path)

    assert product.mission is None
    assert product.instrument is None
    assert product.identification_method == "unknown"


def test_bare_tc_instrument_id_without_selene_mission_is_not_guessed(tmp_path: Path):
    image = _write_image(tmp_path / "generic.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <instrument_id>TC</instrument_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = inspect_product(image, root_dir=tmp_path)

    assert product.instrument is None
    assert product.mission is None


def test_selene_terrain_camera_from_structured_fields(tmp_path: Path):
    image = _write_image(tmp_path / "kaguya_scene.img")
    image.with_suffix(".xml").write_text(
        """<?xml version="1.0"?>
        <Product_Observational>
          <mission_name>SELENE</mission_name>
          <instrument_id>TC</instrument_id>
          <Array_2D_Image>
            <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
            <Axis_Array><elements>4</elements></Axis_Array>
          </Array_2D_Image>
        </Product_Observational>""",
        encoding="utf-8",
    )

    product = inspect_product(image, root_dir=tmp_path)

    assert product.mission == "SELENE"
    assert product.instrument == "TC"
    assert product.identification_method == "pds4_metadata"
