import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import load_json, save_json


WHITESPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")

NAME_LIKE_KEYS = {
    "name",
    "person",
    "title",
    "guarantor",
    "course_coordinator",
    "lecturer",
    "instructor",
    "authors",
    "team_members",
    "organization",
}

PARAGRAPH_KEYS = {
    "about",
    "overview",
    "abstract",
    "details",
    "description",
    "content",
    "current_research_topics",
}


def normalise_line(line: str) -> str:
    """Normalise a single text line while preserving visible content."""
    line = unicodedata.normalize("NFKC", line)
    line = line.replace("\xa0", " ")
    line = ZERO_WIDTH_RE.sub("", line)
    line = WHITESPACE_RE.sub(" ", line)
    return line.strip()


def normalise_text(text: str) -> str:
    """Normalise free text, preserving line boundaries for list-like content."""
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    normalised_lines = [normalise_line(line) for line in lines]

    # Collapse repeated blank lines, but keep paragraph breaks.
    collapsed = []
    blank_streak = 0
    for line in normalised_lines:
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                collapsed.append(line)
        else:
            blank_streak = 0
            collapsed.append(line)

    return "\n".join(collapsed).strip()


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def is_name_like_key(key: str) -> bool:
    key_norm = key.lower().strip()
    if key_norm in NAME_LIKE_KEYS:
        return True
    return any(token in key_norm for token in ["name", "author", "lectur", "instructor", "guarantor", "coordinator"])


def is_url_key(key: str) -> bool:
    key_norm = key.lower().strip()
    return key_norm == "url" or key_norm.endswith("_url")


def should_join_paragraphs(key: str | None) -> bool:
    if not key:
        return False
    key_norm = key.lower().strip()
    return key_norm in PARAGRAPH_KEYS


def collapse_text_list(items: list[str], parent_key: str | None) -> str | list[str]:
    if len(items) == 1:
        return items[0]
    if should_join_paragraphs(parent_key):
        return "\n\n".join(items)
    return items


def split_atomic_texts(text: str) -> list[str]:
    """Split a text blob into small comparable units for verification."""
    text = normalise_text(text)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n+|\s*\|\s*", text) if p.strip()]
    return parts if parts else [text]


def normalise_obj(
    obj: Any,
    parent_key: str | None = None,
    diacritics_mode: str = "names",
    drop_links: bool = False,
    drop_urls: bool = False,
):
    """Recursively normalise all string values in nested dict/list structures."""
    if isinstance(obj, str):
        text = normalise_text(obj)
        if diacritics_mode == "all":
            text = strip_diacritics(text)
        elif diacritics_mode == "names" and parent_key and is_name_like_key(parent_key):
            text = strip_diacritics(text)
        return text

    if isinstance(obj, list):
        normalised_items = [
            normalise_obj(
                item,
                parent_key=parent_key,
                diacritics_mode=diacritics_mode,
                drop_links=drop_links,
                drop_urls=drop_urls,
            )
            for item in obj
        ]

        # Remove null-ish items after recursive cleanup.
        normalised_items = [item for item in normalised_items if item not in (None, "", [], {})]

        # Collapse list of {"text": "..."} style wrappers to plain strings.
        if normalised_items and all(isinstance(item, dict) and set(item.keys()) <= {"text"} for item in normalised_items):
            texts = [item.get("text", "") for item in normalised_items]
            texts = [t for t in texts if t]
            if not texts:
                return []
            return collapse_text_list(texts, parent_key=parent_key)

        return normalised_items

    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if drop_links and key.lower() == "links":
                continue
            if drop_urls and is_url_key(key):
                continue

            norm_value = normalise_obj(
                value,
                parent_key=key,
                diacritics_mode=diacritics_mode,
                drop_links=drop_links,
                drop_urls=drop_urls,
            )

            if norm_value in (None, "", [], {}):
                continue

            # Remove placeholder detail-only entries.
            if key.lower() == "title" and isinstance(norm_value, str) and norm_value.strip().lower() == "detail":
                continue

            out[key] = norm_value

        # Flatten wrappers like {"value": "..."} or {"text": "..."}.
        if set(out.keys()) == {"value"}:
            return out["value"]

        # Resolve redundant layering like {"about": {"about": ...}}
        if parent_key and parent_key in out:
            duplicated = out.pop(parent_key)
            if duplicated not in (None, "", [], {}):
                if "overview" not in out:
                    out["overview"] = duplicated
                else:
                    # Merge safely if both are strings/lists
                    if isinstance(out["overview"], str) and isinstance(duplicated, str):
                        out["overview"] = f"{out['overview']}\n\n{duplicated}".strip()
                    elif isinstance(out["overview"], list) and isinstance(duplicated, list):
                        out["overview"] = out["overview"] + duplicated

        # Collapse one-key dicts holding only "text".
        if set(out.keys()) == {"text"}:
            return out["text"]

        return out

    return obj


def collect_text_units(
    obj: Any,
    drop_links: bool,
    drop_urls: bool,
    diacritics_mode: str,
    parent_key: str | None = None,
) -> list[str]:
    """Collect comparable text units from an object for source-vs-clean verification."""
    units: list[str] = []

    if isinstance(obj, str):
        text = normalise_text(obj)
        if diacritics_mode == "all":
            text = strip_diacritics(text)
        elif diacritics_mode == "names" and parent_key and is_name_like_key(parent_key):
            text = strip_diacritics(text)
        units.extend(split_atomic_texts(text))
        return units

    if isinstance(obj, list):
        for item in obj:
            units.extend(
                collect_text_units(
                    item,
                    drop_links=drop_links,
                    drop_urls=drop_urls,
                    diacritics_mode=diacritics_mode,
                    parent_key=parent_key,
                )
            )
        return units

    if isinstance(obj, dict):
        for key, value in obj.items():
            if drop_links and key.lower() == "links":
                continue
            if drop_urls and is_url_key(key):
                continue
            if key.lower() == "title" and isinstance(value, str) and normalise_text(value).lower() == "detail":
                continue

            units.extend(
                collect_text_units(
                    value,
                    drop_links=drop_links,
                    drop_urls=drop_urls,
                    diacritics_mode=diacritics_mode,
                    parent_key=key,
                )
            )
        return units

    return units


def verify_file_coverage(
    source_data: Any,
    cleaned_data: Any,
    drop_links: bool,
    drop_urls: bool,
    diacritics_mode: str,
    min_coverage_ratio: float,
) -> tuple[bool, float, int, int]:
    """Verify that cleaned data preserves most source text after intentional drops."""
    src_units = collect_text_units(
        source_data,
        drop_links=drop_links,
        drop_urls=drop_urls,
        diacritics_mode=diacritics_mode,
    )
    clean_units = collect_text_units(
        cleaned_data,
        drop_links=False,
        drop_urls=False,
        diacritics_mode=diacritics_mode,
    )

    src_counter = Counter(src_units)
    clean_counter = Counter(clean_units)

    if not src_counter:
        return True, 1.0, 0, 0

    matched = 0
    total = sum(src_counter.values())
    clean_blob = "\n".join(clean_units)

    for unit, count in src_counter.items():
        if clean_counter[unit] >= count:
            matched += count
        elif unit and unit in clean_blob:
            matched += min(count, 1)

    ratio = matched / total
    return ratio >= min_coverage_ratio, ratio, matched, total


def normalise_json_file(
    input_path: Path,
    output_path: Path,
    diacritics_mode: str,
    drop_links: bool,
    drop_urls: bool,
) -> None:
    data = load_json(str(input_path))
    normalised = normalise_obj(
        data,
        diacritics_mode=diacritics_mode,
        drop_links=drop_links,
        drop_urls=drop_urls,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(normalised, str(output_path))


def normalise_for_mode(text: str, diacritics_mode: str) -> str:
    out = normalise_text(text)
    if diacritics_mode == "all":
        out = strip_diacritics(out)
    return out


def extract_course_name_map(input_root: Path, diacritics_mode: str) -> dict[str, str]:
    """Build mapping {course_code: full_course_name} from programme curriculum files."""
    code_to_name: dict[str, str] = {}
    program_files = ["MIT-EN.json", "MITAI.json"]

    for fname in program_files:
        path = input_root / fname
        if not path.exists():
            continue

        try:
            data = load_json(str(path))
        except Exception:
            continue

        curriculum = data.get("curriculum", [])
        if not isinstance(curriculum, list):
            continue

        for sem in curriculum:
            if not isinstance(sem, dict):
                continue
            courses = sem.get("courses", [])
            if not isinstance(courses, list):
                continue

            for c in courses:
                if not isinstance(c, dict):
                    continue
                code = c.get("abbreviation")
                title = c.get("title")
                if not code or not title:
                    continue

                code_norm = normalise_for_mode(str(code), diacritics_mode)
                title_norm = normalise_for_mode(str(title), diacritics_mode)
                if code_norm and title_norm:
                    # Keep first encountered canonical title.
                    code_to_name.setdefault(code_norm, title_norm)

    return code_to_name


def iter_json_files(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Create a normalised-text copy of extracted FIT JSON data with the same folder structure."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="extracted_data/fit",
        help="Path to input JSON root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_data_clean/fit",
        help="Path to output root directory for normalised JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional safety cap for number of files to process.",
    )
    parser.add_argument(
        "--diacritics_mode",
        choices=["none", "names", "all"],
        default="names",
        help="How to strip diacritics: 'none' keeps all, 'names' strips name-like fields, 'all' strips all string fields.",
    )
    parser.add_argument(
        "--keep_links",
        action="store_true",
        help="Keep nested 'links' arrays (default removes them in the cleaned output).",
    )
    parser.add_argument(
        "--keep_urls",
        action="store_true",
        help="Keep URL fields like 'url' and '*_url' (default removes them in the cleaned output).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run source-vs-clean verification and report files with low text coverage.",
    )
    parser.add_argument(
        "--min_coverage_ratio",
        type=float,
        default=0.95,
        help="Minimum acceptable text coverage ratio for verification mode.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    processed = 0
    skipped = 0
    verified = 0
    verify_failures = []
    course_name_map = extract_course_name_map(input_root, diacritics_mode=args.diacritics_mode)

    for in_file in iter_json_files(input_root):
        rel_path = in_file.relative_to(input_root)
        out_file = output_root / rel_path

        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue

        source_data = load_json(str(in_file))
        cleaned_data = normalise_obj(
            source_data,
            diacritics_mode=args.diacritics_mode,
            drop_links=not args.keep_links,
            drop_urls=not args.keep_urls,
        )

        # Enrich course files with full course name from programme curriculum mapping.
        if rel_path.parts and rel_path.parts[0] == "courses" and isinstance(cleaned_data, dict):
            metadata = cleaned_data.get("metadata")
            if isinstance(metadata, dict):
                code = metadata.get("code")
                if isinstance(code, str):
                    course_name = course_name_map.get(normalise_for_mode(code, args.diacritics_mode))
                    if course_name:
                        metadata["course_name"] = course_name

        out_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(cleaned_data, str(out_file))

        if args.verify:
            ok, ratio, matched, total = verify_file_coverage(
                source_data,
                cleaned_data,
                drop_links=not args.keep_links,
                drop_urls=not args.keep_urls,
                diacritics_mode=args.diacritics_mode,
                min_coverage_ratio=args.min_coverage_ratio,
            )
            verified += 1
            if not ok:
                verify_failures.append((str(rel_path), ratio, matched, total))

        processed += 1

        if args.max_files is not None and processed >= args.max_files:
            break

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Processed:   {processed}")
    print(f"Skipped:     {skipped}")
    if args.verify:
        print(f"Verified:    {verified}")
        print(f"Failures:    {len(verify_failures)}")
        if verify_failures:
            print("Top verification failures (relative_path, coverage, matched/total):")
            for rel_path, ratio, matched, total in sorted(verify_failures, key=lambda x: x[1])[:20]:
                print(f"- {rel_path} | {ratio:.3f} | {matched}/{total}")


if __name__ == "__main__":
    main()
