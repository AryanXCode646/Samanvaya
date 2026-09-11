from datetime import timezone
from pathlib import Path

from lunar_core.data_io.archive_status import MissionArchiveStatus, check_archive, latest_product, status_with_local_products
from lunar_core.data_io.mission_product import MissionProduct


def product(product_id: str, timestamp: str, instrument: str = "OHRC") -> MissionProduct:
    return MissionProduct(
        mission="Chandrayaan-2",
        instrument=instrument,
        product_id=product_id,
        image_path=Path(f"{product_id}.tif"),
        acquisition_time=timestamp,
        status="validated",
    )


def test_latest_product_uses_acquisition_timestamp_not_filename():
    products = [
        product("zzz_old_name", "2025-01-01T00:00:00Z"),
        product("aaa_new_name", "2025-06-01T00:00:00Z"),
    ]
    assert latest_product(products).product_id == "aaa_new_name"


def test_latest_product_filters_instrument():
    products = [
        product("ohrc", "2025-01-01T00:00:00Z", "OHRC"),
        product("tmc", "2025-06-01T00:00:00Z", "TMC-2"),
    ]
    assert latest_product(products, instrument="OHRC").product_id == "ohrc"


def test_archive_network_failure_is_safe():
    def failing_opener(*args, **kwargs):
        raise OSError("offline")

    status = check_archive(opener=failing_opener)
    assert status.availability == "NETWORK_ERROR"
    assert status.latest_product_id is None


def test_archive_status_preserves_local_product_provenance():
    status = MissionArchiveStatus(
        source="ISRO / ISSDC",
        mission="Chandrayaan-2",
        last_checked="2026-09-11T00:00:00+00:00",
        latest_product_id=None,
        latest_acquisition_time=None,
        availability="LOGIN_REQUIRED",
        access_mode="official archive",
        message="reachable",
        official_url="https://pradan.issdc.gov.in/ch2/",
    )
    updated = status_with_local_products(status, [product("local", "2026-01-01T00:00:00Z")])
    assert updated.availability == "AVAILABLE_LOCAL"
    assert updated.latest_product_id == "local"
    assert updated.access_mode == "authorized local product"
