from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from utils.common import load_json

_SEMESTER_WINTER = re.compile(r"\bwinter\b", re.IGNORECASE)
_SEMESTER_SUMMER = re.compile(r"\bsummer\b", re.IGNORECASE)
_PERSON_TITLE_TOKENS = {
    "bc",
    "bsc",
    "csc",
    "dr",
    "eng",
    "h",
    "hc",
    "ing",
    "mba",
    "mgr",
    "msc",
    "phd",
    "prof",
    "rndr",
}


@dataclass(slots=True)
class PersonContactRecord:
    profile_id: str
    name: str | None
    name_normalized: str
    email: str | None
    office_room: str | None
    has_email: bool
    has_office_room: bool
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CourseSemesterRecord:
    course_id: str
    code: str | None
    course_name: str | None
    semester_label: str | None
    semester_norm: str
    has_semester: bool
    semester_source_field: str | None
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_text(text: str) -> str:
    """Normalize noisy text for matching (NFKC, lowercase, no diacritics, compact spaces)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().strip()
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = re.sub(r"[^\w\s-]", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def canonicalize_person_name(name_like: str) -> str:
    """Canonical person name without degree/title tokens, e.g. 'Chudý_Peter,_doc._Ing.' -> 'chudy peter'."""
    if not name_like:
        return ""

    raw = unicodedata.normalize("NFKC", name_like).replace("_", " ").strip()
    # Most filenames use the pattern: Surname_Name,_titles... ; keep only the name segment.
    raw = raw.split(",", 1)[0].strip()

    canonical = canonicalize_text(raw)
    tokens = [token for token in canonical.split() if token not in _PERSON_TITLE_TOKENS]
    return " ".join(tokens).strip()


def normalize_semester_label(label: str | None) -> str:
    if not label:
        return "unknown"
    if _SEMESTER_WINTER.search(label):
        return "winter"
    if _SEMESTER_SUMMER.search(label):
        return "summer"
    return "unknown"


def _safe_text(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def extract_person_contact_record(data: dict[str, Any], source_path: str) -> PersonContactRecord:
    contact = data.get("contact") if isinstance(data, dict) else None
    details = contact.get("details") if isinstance(contact, dict) else None

    email = _safe_text(details.get("e_mail")) if isinstance(details, dict) else None
    office = _safe_text(details.get("room")) if isinstance(details, dict) else None
    name = _safe_text(data.get("name"))

    name_for_id = name if name else Path(source_path).stem
    name_normalized = canonicalize_person_name(name_for_id)
    profile_id = name_normalized
    return PersonContactRecord(
        profile_id=profile_id,
        name=name,
        name_normalized=name_normalized,
        email=email,
        office_room=office,
        has_email=email is not None,
        has_office_room=office is not None,
        source_path=source_path,
    )


def _derive_semester_from_metadata(metadata: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """
    Returns (semester_label, semester_norm, source_field).

    Uses fallbacks because some records have shifted metadata values.
    """
    candidates = [
        ("semester", _safe_text(metadata.get("semester"))),
        ("credits", _safe_text(metadata.get("credits"))),
        ("academic_year", _safe_text(metadata.get("academic_year"))),
    ]

    first_label = None
    for field, value in candidates:
        if value and first_label is None:
            first_label = value

        norm = normalize_semester_label(value)
        if norm != "unknown":
            return value, norm, field

    return first_label, "unknown", "semester" if first_label else None


def extract_course_semester_record(data: dict[str, Any], source_path: str) -> CourseSemesterRecord:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}

    code = _safe_text(metadata.get("code"))
    course_name = _safe_text(metadata.get("course_name"))
    semester_label, semester_norm, source_field = _derive_semester_from_metadata(metadata)

    course_id = canonicalize_text(code) if code else canonicalize_text(Path(source_path).stem)
    return CourseSemesterRecord(
        course_id=course_id,
        code=code,
        course_name=course_name,
        semester_label=semester_label,
        semester_norm=semester_norm,
        has_semester=semester_norm != "unknown",
        semester_source_field=source_field,
        source_path=source_path,
    )


def scan_cleaned_contact_semester_coverage(clean_root: Path) -> dict[str, Any]:
    people_dir = clean_root / "personnel_profiles"
    courses_dir = clean_root / "courses"

    person_files = sorted(people_dir.glob("*.json"))
    course_files = sorted(courses_dir.glob("*.json"))

    people_records: list[PersonContactRecord] = []
    for path in person_files:
        people_records.append(extract_person_contact_record(load_json(str(path)), str(path)))

    course_records: list[CourseSemesterRecord] = []
    for path in course_files:
        course_records.append(extract_course_semester_record(load_json(str(path)), str(path)))

    missing_email = [Path(r.source_path).name for r in people_records if not r.has_email]
    missing_room = [Path(r.source_path).name for r in people_records if not r.has_office_room]
    unknown_semester = [Path(r.source_path).name for r in course_records if not r.has_semester]

    return {
        "personnel_total": len(people_records),
        "personnel_with_email": sum(1 for r in people_records if r.has_email),
        "personnel_with_office_room": sum(1 for r in people_records if r.has_office_room),
        "courses_total": len(course_records),
        "courses_with_semester": sum(1 for r in course_records if r.has_semester),
        "missing_email_profiles": missing_email,
        "missing_room_profiles": missing_room,
        "unknown_semester_courses": unknown_semester,
    }
