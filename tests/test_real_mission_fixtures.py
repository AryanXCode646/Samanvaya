from pathlib import Path

from lunar_core.data_io.mission_adapters import adapter_for_mission
from lunar_core.data_io.mission_catalog import inspect_product


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
