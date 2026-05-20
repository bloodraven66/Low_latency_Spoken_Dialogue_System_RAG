from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import load_json
from utils.kg_contract import canonicalize_person_name, canonicalize_text

TOKEN_RE = re.compile(r"\S+")


@dataclass(slots=True)
class ChunkingConfig:
    chunk_impl: str = "basic"
    chunk_size: int = 384
    chunk_overlap: int = 64
    min_tokens: int = 20
    include_metadata_header: bool = False
    v2_short_field_min_tokens: int = 1
    v2_drop_noisy_fields: bool = True
    context_max_tokens: int = 48


@dataclass(slots=True)
class FieldTextRecord:
    entity_type: str
    entity_id: str
    source_path: str
    source_field: str
    text: str
    entity_context: str = ""


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    entity_type: str
    entity_id: str
    source_path: str
    source_field: str
    chunk_index: int
    token_count: int
    chunk_text: str
    embedding_text: str


_SHORT_FACT_FIELD_PATTERNS = [
    re.compile(r"^contact\.details\.(e_mail|room)$"),
    re.compile(r"^content\.language_of_instruction$"),
    re.compile(r"^metadata\.(code|course_name|semester)$"),
    re.compile(r"^content\.(lecturer|instructor|who_teaches|contact_person|code_expansion)$"),
    re.compile(r"^metadata\.agency$"),
    re.compile(r"^sections\.(type|publisher)\.text$"),
    re.compile(r"^year$"),
]

_NOISY_FIELD_PATTERNS = [
    re.compile(r"^publicationresults\[\d+\]\.citation$"),
    re.compile(r"^sections\.publication_results\.text$"),
]


def _is_short_fact_field(source_field: str) -> bool:
    return any(p.search(source_field) for p in _SHORT_FACT_FIELD_PATTERNS)


def _is_noisy_field(source_field: str) -> bool:
    return any(p.search(source_field) for p in _NOISY_FIELD_PATTERNS)


def _effective_min_tokens(rec: FieldTextRecord, config: ChunkingConfig) -> int:
    if config.chunk_impl not in {"basic_v2", "basic_ctx"}:
        return config.min_tokens
    if _is_short_fact_field(rec.source_field):
        return max(1, int(config.v2_short_field_min_tokens))
    return config.min_tokens


def _sanitize_name_token(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "unknown"


def build_embedding_data_name(model_name: str, config: ChunkingConfig) -> str:
    model_token = _sanitize_name_token(model_name)
    header_token = "hdr1" if config.include_metadata_header else "hdr0"
    return (
        f"{model_token}"
        f"__cs{config.chunk_size}"
        f"_ov{config.chunk_overlap}"
        f"_min{config.min_tokens}"
        f"_{header_token}"
    )


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _windowed_chunks(tokens: list[str], chunk_size: int, chunk_overlap: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not tokens:
        return []

    if len(tokens) <= chunk_size:
        return [tokens]

    stride = chunk_size - chunk_overlap
    out: list[list[str]] = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        if not window:
            continue
        out.append(window)
        if start + chunk_size >= len(tokens):
            break
    return out


def default_header(record: FieldTextRecord) -> str:
    return f"[ENTITY={record.entity_type}] [ID={record.entity_id}] [FIELD={record.source_field}]"


def _iter_string_fields(obj: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, str):
        value = obj.strip()
        if value:
            yield (prefix or "text", value)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_string_fields(value, child)
        return

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_string_fields(value, child)


def _entity_type_from_relative_path(rel: Path) -> str:
    top = rel.parts[0] if len(rel.parts) > 1 else "root"
    mapping = {
        "personnel_profiles": "Person",
        "courses": "Course",
        "groups": "ResearchGroup",
        "projects": "Project",
        "publications": "Publication",
    }
    return mapping.get(top, "Document")


def _entity_id_from_file(data: dict[str, Any], rel: Path) -> str:
    top = rel.parts[0] if len(rel.parts) > 1 else "root"
    stem = rel.stem

    if top == "personnel_profiles":
        return canonicalize_person_name(stem)
    if top == "courses":
        metadata = data.get("metadata") if isinstance(data, dict) else None
        code = metadata.get("code") if isinstance(metadata, dict) else None
        if isinstance(code, str) and code.strip():
            return canonicalize_text(code)
        return canonicalize_text(stem)
    if top == "groups":
        name = data.get("group") if isinstance(data, dict) else None
        return canonicalize_text(name) if isinstance(name, str) and name.strip() else canonicalize_text(stem)
    if top == "projects":
        title = data.get("title") if isinstance(data, dict) else None
        if isinstance(title, str) and title.strip():
            return canonicalize_text(title)
        project = data.get("project") if isinstance(data, dict) else None
        if isinstance(project, str) and project.strip():
            return canonicalize_text(project)
        return canonicalize_text(stem)
    if top == "publications":
        title = data.get("title") if isinstance(data, dict) else None
        return canonicalize_text(title) if isinstance(title, str) and title.strip() else canonicalize_text(stem)

    return canonicalize_text(stem)


def _safe_get(data: dict[str, Any], path: str) -> str | None:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if isinstance(cur, str):
        val = cur.strip()
        return val if val else None
    return None


def _first_list_name(data: dict[str, Any], path: str) -> str | None:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if isinstance(cur, list) and cur:
        head = cur[0]
        if isinstance(head, dict):
            name = head.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(head, str) and head.strip():
            return head.strip()
    return None


def _build_entity_context(*, entity_type: str, entity_id: str, data: dict[str, Any], rel: Path) -> str:
    parts: list[str] = [f"entity_type={entity_type}", f"entity_id={entity_id}"]

    if entity_type == "Course":
        code = _safe_get(data, "metadata.code") or rel.stem
        cname = _safe_get(data, "metadata.course_name")
        dept = _safe_get(data, "content.department")
        parts.append(f"code={code}")
        if cname:
            parts.append(f"course_name={cname}")
        if dept:
            parts.append(f"department={dept}")
    elif entity_type == "Person":
        email = _safe_get(data, "contact.details.e_mail")
        room = _safe_get(data, "contact.details.room")
        parts.append(f"person={rel.stem}")
        if email:
            parts.append(f"email={email}")
        if room:
            parts.append(f"room={room}")
    elif entity_type == "ResearchGroup":
        group = data.get("group") if isinstance(data.get("group"), str) else rel.stem
        contact = _first_list_name(data, "team.principal_researcher")
        parts.append(f"group={str(group).strip()}")
        if contact:
            parts.append(f"contact={contact}")
    elif entity_type == "Project":
        title = _safe_get(data, "title") or _safe_get(data, "project") or rel.stem
        code = _safe_get(data, "metadata.code")
        agency = _safe_get(data, "metadata.agency")
        parts.append(f"project={title}")
        if code:
            parts.append(f"code={code}")
        if agency:
            parts.append(f"agency={agency}")
    elif entity_type == "Publication":
        title = _safe_get(data, "title") or rel.stem
        year = _safe_get(data, "year")
        ptype = _safe_get(data, "sections.type.text")
        parts.append(f"title={title}")
        if year:
            parts.append(f"year={year}")
        if ptype:
            parts.append(f"type={ptype}")

    return " | ".join(parts)


def iter_field_text_records(clean_root: Path) -> Iterable[FieldTextRecord]:
    for path in sorted(clean_root.rglob("*.json")):
        if not path.is_file():
            continue
        data = load_json(str(path))
        rel = path.relative_to(clean_root)
        entity_type = _entity_type_from_relative_path(rel)
        entity_id = _entity_id_from_file(data, rel)
        entity_context = _build_entity_context(entity_type=entity_type, entity_id=entity_id, data=data, rel=rel)

        for source_field, text in _iter_string_fields(data):
            yield FieldTextRecord(
                entity_type=entity_type,
                entity_id=entity_id,
                source_path=str(path),
                source_field=source_field,
                text=text,
                entity_context=entity_context,
            )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    toks = _tokenize(text)
    if max_tokens <= 0 or len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens])


def _context_prefix(rec: FieldTextRecord, config: ChunkingConfig) -> str:
    if config.chunk_impl != "basic_ctx":
        return ""
    ctx = _truncate_to_tokens(rec.entity_context or "", config.context_max_tokens).strip()
    if not ctx:
        return ""
    return f"[CONTEXT] {ctx} [FIELD={rec.source_field}]"


def build_chunks(
    records: Iterable[FieldTextRecord],
    config: ChunkingConfig,
    header_builder=default_header,
) -> list[ChunkRecord]:
    if config.chunk_impl not in {"basic", "basic_v2", "basic_ctx"}:
        raise ValueError(f"Unsupported chunk_impl: {config.chunk_impl}")

    chunks: list[ChunkRecord] = []

    for rec in records:
        if config.chunk_impl in {"basic_v2", "basic_ctx"} and config.v2_drop_noisy_fields and _is_noisy_field(rec.source_field):
            continue

        tokens = _tokenize(rec.text)
        min_tokens = _effective_min_tokens(rec, config)
        if len(tokens) < min_tokens:
            continue

        windows = _windowed_chunks(tokens, config.chunk_size, config.chunk_overlap)
        for i, window in enumerate(windows):
            chunk_text = " ".join(window).strip()
            if not chunk_text:
                continue

            context_prefix = _context_prefix(rec, config)
            if context_prefix:
                chunk_text = f"{context_prefix}\n{chunk_text}".strip()

            if config.include_metadata_header:
                header = header_builder(rec)
                embedding_text = f"{header}\n{chunk_text}" if header else chunk_text
            else:
                embedding_text = chunk_text

            chunk_id = f"{rec.entity_type.lower()}:{rec.entity_id}:{rec.source_field}:{i}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    entity_type=rec.entity_type,
                    entity_id=rec.entity_id,
                    source_path=rec.source_path,
                    source_field=rec.source_field,
                    chunk_index=i,
                    token_count=len(window),
                    chunk_text=chunk_text,
                    embedding_text=embedding_text,
                )
            )

    return chunks


def write_jsonl(records: Iterable[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def _build_generation_config_payload(
    *,
    clean_root: Path,
    output_path: Path,
    config: ChunkingConfig,
    summary: dict[str, Any],
    embedding_data_name: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(existing or {})
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["data_root"] = str(clean_root)
    payload["embedding_data_name"] = embedding_data_name
    payload["artifacts"] = {
        **(payload.get("artifacts") or {}),
        "chunks_jsonl": str(output_path),
    }
    payload["chunking"] = {
        "algorithm": "token_window",
        **asdict(config),
    }
    payload["stats"] = {
        **(payload.get("stats") or {}),
        "chunking": summary,
    }
    payload.setdefault("embedding", {})
    payload.setdefault("index", {})
    return payload


def write_generation_config(
    *,
    clean_root: Path,
    output_path: Path,
    config_output_path: Path,
    chunking_config: ChunkingConfig,
    summary: dict[str, Any],
    embedding_data_name: str,
) -> dict[str, Any]:
    existing = None
    if config_output_path.is_file():
        existing = load_json(str(config_output_path))

    payload = _build_generation_config_payload(
        clean_root=clean_root,
        output_path=output_path,
        config=chunking_config,
        summary=summary,
        embedding_data_name=embedding_data_name,
        existing=existing if isinstance(existing, dict) else None,
    )
    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _summary(chunks: list[ChunkRecord]) -> dict[str, Any]:
    by_entity_type: dict[str, int] = {}
    by_field: dict[str, int] = {}

    for c in chunks:
        by_entity_type[c.entity_type] = by_entity_type.get(c.entity_type, 0) + 1
        by_field[c.source_field] = by_field.get(c.source_field, 0) + 1

    top_fields = sorted(by_field.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "chunks_total": len(chunks),
        "entity_type_counts": by_entity_type,
        "top_source_fields": top_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build default vector chunks from cleaned FIT JSON (modular chunking pipeline).")
    parser.add_argument("--clean_root", type=str, default="extracted_data_clean/fit")
    parser.add_argument("--embeddings_root", type=str, default="embeddings")
    parser.add_argument("--embedding_model_name", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument(
        "--embedding_data_name",
        type=str,
        default=None,
        help="Artifact folder name under embeddings_root. If omitted, derived from model + chunk params.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit chunks output path. If omitted, uses embeddings_root/embedding_data_name/chunks.jsonl",
    )
    parser.add_argument("--chunk_size", type=int, default=384)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--min_tokens", type=int, default=20)
    parser.add_argument(
        "--chunk_impl",
        type=str,
        choices=["basic", "basic_v2", "basic_ctx"],
        default="basic",
        help="Chunking profile. basic_v2 enables short-fact retention/noisy-field suppression; basic_ctx adds compact entity context into chunk_text.",
    )
    parser.add_argument(
        "--v2_short_field_min_tokens",
        type=int,
        default=1,
        help="For basic_v2 only: minimum token threshold for short factual fields (e.g., email, code).",
    )
    parser.add_argument(
        "--v2_keep_noisy_fields",
        action="store_true",
        help="For basic_v2 only: keep noisy citation-heavy fields (default is to drop them).",
    )
    parser.add_argument(
        "--config_output",
        type=str,
        default=None,
        help="Path to generation config JSON. Defaults to <output_dir>/generation_config.json",
    )
    parser.add_argument(
        "--include_metadata_header",
        action="store_true",
        help="Prepend a compact metadata header to embedding_text (chunk_text stays clean).",
    )
    parser.add_argument(
        "--context_max_tokens",
        type=int,
        default=48,
        help="For basic_ctx only: max tokens used for injected entity context prefix.",
    )
    args = parser.parse_args()

    config = ChunkingConfig(
        chunk_impl=args.chunk_impl,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_tokens=args.min_tokens,
        include_metadata_header=args.include_metadata_header,
        v2_short_field_min_tokens=int(args.v2_short_field_min_tokens),
        v2_drop_noisy_fields=not bool(args.v2_keep_noisy_fields),
        context_max_tokens=int(args.context_max_tokens),
    )

    clean_root = Path(args.clean_root).resolve()
    embedding_data_name = args.embedding_data_name or build_embedding_data_name(args.embedding_model_name, config)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = (Path(args.embeddings_root).resolve() / embedding_data_name / "chunks.jsonl")

    config_output_path = Path(args.config_output).resolve() if args.config_output else out_path.parent / "generation_config.json"

    field_records = list(iter_field_text_records(clean_root))
    chunks = build_chunks(field_records, config=config)
    write_jsonl(chunks, out_path)
    summary = _summary(chunks)
    generation_config = write_generation_config(
        clean_root=clean_root,
        output_path=out_path,
        config_output_path=config_output_path,
        chunking_config=config,
        summary=summary,
        embedding_data_name=embedding_data_name,
    )

    print("[chunk_build]")
    print(json.dumps({
        "clean_root": str(clean_root),
        "output": str(out_path),
    "embedding_data_name": embedding_data_name,
        "config_output": str(config_output_path),
        "config": asdict(config),
        **summary,
        "generation_config_sections": sorted(generation_config.keys()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
