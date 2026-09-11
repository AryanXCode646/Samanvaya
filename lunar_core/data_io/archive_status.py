"""Truthful mission-archive status and local product selection helpers.

This module deliberately does not authenticate, scrape protected pages, or download
mission rasters. Official archive landing-page reachability and authorized local
products are separate facts exposed to the UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lunar_core.data_io.mission_catalog import scan
from lunar_core.data_io.mission_product import MissionProduct

OFFICIAL_ARCHIVES = {
    "Chandrayaan-2 PRADAN": "https://pradan.issdc.gov.in/ch2/",
    "Chandrayaan Data Explorer": "https://chmapbrowse.issdc.gov.in/",
    "LRO Search": "https://data.lroc.im-ldi.com/lroc/search",
    "SELENE": "https://darts.isas.jaxa.jp/app/pdap/selene/",
}


@dataclass(frozen=True)
class MissionArchiveStatus:
    source: str
    mission: str
    last_checked: str
    latest_product_id: Optional[str]
    latest_acquisition_time: Optional[str]
    availability: str
    access_mode: str
    message: str
    official_url: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_acquisition_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def latest_product(products: Iterable[MissionProduct], instrument: Optional[str] = None) -> Optional[MissionProduct]:
    """Return the newest timestamped product; never use filename ordering."""
    candidates = [
        product
        for product in products
        if product.status in {"validated", "partial", "parsed"}
        and product.acquisition_time
        and (instrument is None or (product.instrument or "").upper() == instrument.upper())
    ]
    candidates.sort(key=lambda product: _parse_acquisition_time(product.acquisition_time) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return candidates[0] if candidates else None


def scan_local_chandrayaan2(root: Path) -> list[MissionProduct]:
    """Scan authorized local products without loading raster pixels."""
    from lunar_core.data_io.mission_adapters import Chandrayaan2Adapter

    return Chandrayaan2Adapter().scan(root)


def check_archive(
    mission: str = "Chandrayaan-2",
    official_url: str = OFFICIAL_ARCHIVES["Chandrayaan-2 PRADAN"],
    opener: Callable[..., object] = urlopen,
    timeout_seconds: float = 8.0,
) -> MissionArchiveStatus:
    """Check official archive reachability without claiming product freshness."""
    checked = utc_now()
    request = Request(official_url, headers={"User-Agent": "Samanvaya-archive-status/1.0"}, method="HEAD")
    try:
        response = opener(request, timeout=timeout_seconds)
        status_code = getattr(response, "status", 200)
        if status_code >= 400:
            raise HTTPError(official_url, status_code, "archive returned an error", None, None)
        return MissionArchiveStatus(
            source="ISRO / ISSDC",
            mission=mission,
            last_checked=checked,
            latest_product_id=None,
            latest_acquisition_time=None,
            availability="LOGIN_REQUIRED",
            access_mode="official archive; user authentication may be required",
            message="Official archive is reachable. Product discovery/download remains subject to ISSDC access.",
            official_url=official_url,
        )
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return MissionArchiveStatus(
            source="ISRO / ISSDC",
            mission=mission,
            last_checked=checked,
            latest_product_id=None,
            latest_acquisition_time=None,
            availability="NETWORK_ERROR",
            access_mode="local authorized data only",
            message=f"Archive check failed: {type(exc).__name__}",
            official_url=official_url,
        )


def status_with_local_products(
    archive_status: MissionArchiveStatus,
    products: Iterable[MissionProduct],
    instrument: Optional[str] = None,
) -> MissionArchiveStatus:
    """Attach the newest locally authorized product while preserving archive facts."""
    product = latest_product(products, instrument=instrument)
    if product is None:
        return archive_status
    return MissionArchiveStatus(
        **{
            **archive_status.to_dict(),
            "latest_product_id": product.product_id,
            "latest_acquisition_time": product.acquisition_time,
            "availability": "AVAILABLE_LOCAL",
            "access_mode": "authorized local product",
            "message": "Newest timestamped local product is available for registration.",
        }
    )
