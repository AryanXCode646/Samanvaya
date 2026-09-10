"""Canonical PDS4 label association and constrained mission/instrument identification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from lunar_core.data_io.mission_product import IdentificationMethod


class LabelAssociationStatus(str, Enum):
    RESOLVED = "resolved"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


class LabelAssociationMethod(str, Enum):
    SAME_STEM = "same_stem"
    EXPLICIT_FILE_REFERENCE = "explicit_file_reference"
    MISSION_CONVENTION = "mission_convention"


@dataclass(frozen=True)
class LabelResolution:
    """Deterministic result of associating a detached product with a PDS4 label."""

    path: Optional[Path]
    status: LabelAssociationStatus
    method: Optional[LabelAssociationMethod] = None
    message: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.status is LabelAssociationStatus.RESOLVED and self.path is not None


_FILE_NAME_TAGS = frozenset({"file_name", "filename", "data_file_name"})
_MISSION_FIELD_TAGS = (
    "mission_name",
    "mission_id",
    "mission",
    "spacecraft_id",
    "spacecraft_name",
    "spacecraft",
)
_INSTRUMENT_FIELD_TAGS = (
    "instrument_name",
    "instrument_id",
    "instrument",
    "sensor_name",
    "sensor_id",
    "camera_name",
    "instrument_type",
)
_PRODUCT_ID_TAGS = ("product_id", "product_identifier", "logical_identifier", "product_name")

# Long names may appear inside a dedicated instrument field. Short tokens must not
# be matched as generic substrings of arbitrary label text.
_INSTRUMENT_EXACT = {
    "OHRC": "OHRC",
    "OHR": "OHRC",
    "TMC-2": "TMC-2",
    "TMC2": "TMC-2",
    "TMC_2": "TMC-2",
    "TMC": "TMC-2",
    "IIRS": "IIRS",
    "HYSI": "HYSI",
    "NAC": "NAC",
    "LROC": "LROC",
    "LROC-NAC": "NAC",
    "LROC_NAC": "NAC",
    "TERRAIN CAMERA": "TC",
    "TERRAIN MAPPING CAMERA": "TMC-2",
    "TERRAIN MAPPING CAMERA-2": "TMC-2",
    "HIGH RESOLUTION CAMERA": "OHRC",
    "ORBITE HIGH RESOLUTION CAMERA": "OHRC",
    "ORBITER HIGH RESOLUTION CAMERA": "OHRC",
    "NARROW ANGLE CAMERA": "NAC",
    "IMAGING INFRARED SPECTROMETER": "IIRS",
    "HYPER SPECTRAL IMAGER": "HYSI",
}

_MISSION_EXACT = {
    "CHANDRAYAAN-1": "Chandrayaan-1",
    "CHANDRAYAAN 1": "Chandrayaan-1",
    "CHANDRAYAAN1": "Chandrayaan-1",
    "CH1": "Chandrayaan-1",
    "CHANDRAYAAN-2": "Chandrayaan-2",
    "CHANDRAYAAN 2": "Chandrayaan-2",
    "CHANDRAYAAN2": "Chandrayaan-2",
    "CH2": "Chandrayaan-2",
    "LRO": "LRO",
    "LROC": "LRO",
    "LUNAR RECONNAISSANCE ORBITER": "LRO",
    "SELENE": "SELENE",
    "KAGUYA": "SELENE",
}

# Authoritative product-id / filename patterns. Bare "TC" is intentionally absent.
_MISSION_PRODUCT_PATTERNS: tuple[tuple[re.Pattern[str], str, Optional[str]], ...] = (
    (re.compile(r"(^|[^A-Z0-9])CH1[_-]?HYS"), "Chandrayaan-1", "HYSI"),
    (re.compile(r"(^|[^A-Z0-9])CH1([^A-Z0-9]|$)"), "Chandrayaan-1", None),
    (re.compile(r"(^|[^A-Z0-9])CH2[_-]?OHR"), "Chandrayaan-2", "OHRC"),
    (re.compile(r"(^|[^A-Z0-9])CH2[_-]?(TMC2|TMC)"), "Chandrayaan-2", "TMC-2"),
    (re.compile(r"(^|[^A-Z0-9])CH2[_-]?IIR"), "Chandrayaan-2", "IIRS"),
    (re.compile(r"(^|[^A-Z0-9])CH2([^A-Z0-9]|$)"), "Chandrayaan-2", None),
    (re.compile(r"(^|[^A-Z0-9])(LROC[_-]?NAC|LRO[_-]?NAC|NAC)([^A-Z0-9]|$)"), "LRO", "NAC"),
    (re.compile(r"(^|[^A-Z0-9])LROC([^A-Z0-9]|$)"), "LRO", "LROC"),
    (re.compile(r"(^|[^A-Z0-9])LRO([^A-Z0-9]|$)"), "LRO", None),
    (re.compile(r"(^|[^A-Z0-9])(TCO|TC1|TC2)([^A-Z0-9]|$)"), "SELENE", "TC"),
    (re.compile(r"(^|[^A-Z0-9])(SELENE|KAGUYA)([^A-Z0-9]|$)"), "SELENE", None),
)


def _tag_name(node: Any) -> str:
    tag = getattr(node, "tag", "")
    if not isinstance(tag, str):
        return str(tag).lower()
    return tag.rsplit("}", 1)[-1].lower()


def _local_values(root: Any, name: str) -> list[str]:
    target = name.lower()
    return [node.text.strip() for node in root.iter() if node.text and _tag_name(node) == target]


def _first_value(root: Any, *names: str) -> Optional[str]:
    for name in names:
        values = _local_values(root, name)
        if values:
            return values[0]
    return None


def _read_label(label_path: Path) -> Any:
    from defusedxml import ElementTree

    return ElementTree.parse(label_path).getroot()


def _normalized_name(path: Path) -> str:
    return path.name.lower()


def _referenced_file_names(root: Any) -> list[str]:
    names: list[str] = []
    for node in root.iter():
        if _tag_name(node) in _FILE_NAME_TAGS and node.text and node.text.strip():
            names.append(Path(node.text.strip()).name.lower())
    return names


def _label_references_image(root: Any, image_path: Path) -> bool:
    referenced = _referenced_file_names(root)
    if not referenced:
        return False
    image_name = _normalized_name(image_path)
    stem = image_path.stem.lower()
    return image_name in referenced or any(Path(name).stem.lower() == stem for name in referenced)


def _same_stem_labels(image_path: Path) -> list[Path]:
    parent = image_path.parent
    if not parent.is_dir():
        return []
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_file() and path.suffix.lower() == ".xml" and path.stem.lower() == image_path.stem.lower()
    )


def _mission_convention_labels(image_path: Path) -> list[Path]:
    parent = image_path.parent
    if not parent.is_dir():
        return []
    stem = image_path.stem
    ordered = [
        parent / f"{stem}_label.xml",
        parent / f"{stem}.lbl.xml",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in ordered:
        matches = [
            path
            for path in parent.iterdir()
            if path.is_file() and path.name.lower() == candidate.name.lower()
        ]
        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                found.append(match)
                seen.add(resolved)
    return found


def _sibling_xml_files(image_path: Path) -> list[Path]:
    parent = image_path.parent
    if not parent.is_dir():
        return []
    return sorted(path for path in parent.iterdir() if path.is_file() and path.suffix.lower() == ".xml")


def _safe_parse(label_path: Path) -> Optional[Any]:
    try:
        return _read_label(label_path)
    except Exception:
        return None


def resolve_product_label(image_path: Path) -> LabelResolution:
    """Resolve one detached PDS4 label using an authoritative, non-guessing policy.

    Association order:
    1. Same-stem XML, when present and not contradicted by an explicit file_name.
    2. Explicit file_name / filename reference from a sibling label.
    3. Mission-specific known label filenames derived from the product stem.
    4. If more than one candidate remains, fail as ambiguous.
    5. If none remain, return missing with an actionable message.

    A directory that happens to contain exactly one XML file is never treated as
    a label unless one of the rules above independently selects it.
    """
    image_path = Path(image_path)
    same_stem = _same_stem_labels(image_path)
    if len(same_stem) > 1:
        names = ", ".join(path.name for path in same_stem)
        return LabelResolution(
            path=None,
            status=LabelAssociationStatus.AMBIGUOUS,
            message=(
                f"Multiple same-stem XML labels for {image_path.name}: {names}. "
                "Provide a single authoritative label or an explicit file_name reference."
            ),
        )
    if len(same_stem) == 1:
        root = _safe_parse(same_stem[0])
        if root is None:
            return LabelResolution(
                path=None,
                status=LabelAssociationStatus.MISSING,
                message=f"Same-stem XML {same_stem[0].name} exists but could not be parsed.",
            )
        referenced = _referenced_file_names(root)
        if referenced and not _label_references_image(root, image_path):
            return LabelResolution(
                path=None,
                status=LabelAssociationStatus.AMBIGUOUS,
                message=(
                    f"Same-stem label {same_stem[0].name} declares file_name {referenced} "
                    f"which does not reference {image_path.name}. Refusing to guess."
                ),
            )
        return LabelResolution(
            path=same_stem[0],
            status=LabelAssociationStatus.RESOLVED,
            method=LabelAssociationMethod.SAME_STEM,
        )

    explicit_matches: list[Path] = []
    for candidate in _sibling_xml_files(image_path):
        root = _safe_parse(candidate)
        if root is not None and _label_references_image(root, image_path):
            explicit_matches.append(candidate)
    if len(explicit_matches) > 1:
        names = ", ".join(path.name for path in explicit_matches)
        return LabelResolution(
            path=None,
            status=LabelAssociationStatus.AMBIGUOUS,
            message=(
                f"Multiple XML labels reference {image_path.name}: {names}. "
                "Refusing to guess among ambiguous labels."
            ),
        )
    if len(explicit_matches) == 1:
        return LabelResolution(
            path=explicit_matches[0],
            status=LabelAssociationStatus.RESOLVED,
            method=LabelAssociationMethod.EXPLICIT_FILE_REFERENCE,
        )

    conventions = _mission_convention_labels(image_path)
    if len(conventions) > 1:
        names = ", ".join(path.name for path in conventions)
        return LabelResolution(
            path=None,
            status=LabelAssociationStatus.AMBIGUOUS,
            message=f"Multiple mission-convention labels for {image_path.name}: {names}.",
        )
    if len(conventions) == 1:
        return LabelResolution(
            path=conventions[0],
            status=LabelAssociationStatus.RESOLVED,
            method=LabelAssociationMethod.MISSION_CONVENTION,
        )

    siblings = [path.name for path in _sibling_xml_files(image_path)]
    if siblings:
        return LabelResolution(
            path=None,
            status=LabelAssociationStatus.MISSING,
            message=(
                f"No authoritative PDS4 label for {image_path.name}. "
                f"XML files present ({', '.join(siblings)}) were not same-stem, "
                "did not declare this file_name, and did not match a known label convention. "
                "A lone XML file in the directory is never selected by default."
            ),
        )
    return LabelResolution(
        path=None,
        status=LabelAssociationStatus.MISSING,
        message=(
            f"Detached product {image_path.name} has no PDS4 XML label. "
            "Place a same-stem .xml file beside the image or declare file_name in the label."
        ),
    )


def _mission_from_structured_value(value: str) -> Optional[str]:
    upper = " ".join(value.upper().split())
    if upper in _MISSION_EXACT:
        return _MISSION_EXACT[upper]
    for key, mission in _MISSION_EXACT.items():
        if len(key) <= 3:
            if re.search(rf"(^|[^A-Z0-9]){re.escape(key)}([^A-Z0-9]|$)", upper):
                return mission
        elif key in upper:
            return mission
    return None


def _instrument_from_structured_value(value: str, *, mission: Optional[str]) -> Optional[str]:
    upper = " ".join(value.upper().split())
    if upper == "TC":
        return "TC" if mission == "SELENE" else None
    if upper in _INSTRUMENT_EXACT:
        return _INSTRUMENT_EXACT[upper]
    for key, instrument in _INSTRUMENT_EXACT.items():
        if len(key) <= 4:
            continue
        if key in upper:
            return instrument
    _, ident_instrument = _identity_from_identifier(upper)
    return ident_instrument


def _identity_from_identifier(identifier: str) -> tuple[Optional[str], Optional[str]]:
    upper = identifier.upper()
    for pattern, mission, instrument in _MISSION_PRODUCT_PATTERNS:
        if pattern.search(upper):
            return mission, instrument
    return None, None


def _structured_mission_values(root: Any) -> list[str]:
    values: list[str] = []
    for name in _MISSION_FIELD_TAGS:
        values.extend(_local_values(root, name))
    for node in root.iter():
        if _tag_name(node) == "investigation_area":
            values.extend(
                child.text.strip()
                for child in node.iter()
                if _tag_name(child) == "name" and child.text
            )
    return values


def _structured_instrument_values(root: Any) -> list[str]:
    values: list[str] = []
    for name in _INSTRUMENT_FIELD_TAGS:
        values.extend(_local_values(root, name))
    for node in root.iter():
        if _tag_name(node) == "observing_system_component":
            component_type = (node.attrib.get("type") or "").lower()
            if component_type and component_type != "instrument":
                continue
            values.extend(
                child.text.strip()
                for child in node.iter()
                if _tag_name(child) in {"name", "description"} and child.text
            )
    return values


def identify_mission_instrument(
    image_path: Path,
    label_root: Any,
    *,
    product_identifier: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], IdentificationMethod]:
    """Identify mission/instrument from structured metadata before any filename heuristic."""
    mission_values = _structured_mission_values(label_root)
    instrument_values = _structured_instrument_values(label_root)
    product_identifier = product_identifier or _first_value(label_root, *_PRODUCT_ID_TAGS)

    mission: Optional[str] = None
    for value in mission_values:
        mission = _mission_from_structured_value(value)
        if mission is not None:
            break

    instrument: Optional[str] = None
    for value in instrument_values:
        instrument = _instrument_from_structured_value(value, mission=mission)
        if instrument is not None:
            break

    if mission is not None or instrument is not None:
        if mission is None and product_identifier:
            mission, _ = _identity_from_identifier(product_identifier)
        return mission, instrument, IdentificationMethod.PDS4_METADATA

    if product_identifier:
        ident_mission, ident_instrument = _identity_from_identifier(product_identifier)
        if ident_mission is not None or ident_instrument is not None:
            if ident_instrument is None:
                for value in instrument_values:
                    ident_instrument = _instrument_from_structured_value(value, mission=ident_mission)
                    if ident_instrument is not None:
                        break
            method = (
                IdentificationMethod.MISSION_SPECIFIC_IDENTIFIER
                if ident_instrument is not None
                else IdentificationMethod.PRODUCT_IDENTIFIER
            )
            return ident_mission, ident_instrument, method
        mission_from_id = _mission_from_structured_value(product_identifier)
        if mission_from_id is not None:
            return mission_from_id, None, IdentificationMethod.PRODUCT_IDENTIFIER

    filename_mission, filename_instrument = _identity_from_identifier(image_path.stem)
    if filename_mission is not None or filename_instrument is not None:
        if filename_instrument is None:
            for value in instrument_values:
                filename_instrument = _instrument_from_structured_value(value, mission=filename_mission)
                if filename_instrument is not None:
                    break
        return filename_mission, filename_instrument, IdentificationMethod.FILENAME_HEURISTIC

    return None, None, IdentificationMethod.UNKNOWN


def identify_from_product_tokens(
    product_id: Optional[str],
    image_stem: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Constrained mission-specific identifier recovery used by adapters."""
    for candidate in (product_id, image_stem):
        if not candidate:
            continue
        mission, instrument = _identity_from_identifier(candidate)
        if mission is not None or instrument is not None:
            return mission, instrument
    return None, None
