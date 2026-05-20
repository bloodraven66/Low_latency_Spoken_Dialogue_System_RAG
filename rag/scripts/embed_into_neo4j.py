from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import load_json
from utils.kg_contract import (
	canonicalize_person_name,
	canonicalize_text,
	extract_course_semester_record,
	extract_person_contact_record,
)


ROLE_SEPARATOR_RE = re.compile(r"\s*\|\s*")
DEPARTMENT_TOKEN_RE = re.compile(r"^\([A-Z0-9\-/]+\)$")
CAPITALIZED_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
MULTI_VALUE_SEP_RE = re.compile(r"\s*[;|]\s*")
FIT_PERSON_UID_RE = re.compile(r"/person/(\d+)/")

EDGE_SOURCE_PRIORITY: dict[str, int] = {
	"free_text_fallback": 1,
	"known_personnel_filename": 2,
	"anchored_identity": 3,
}

EDGE_SOURCE_CONFIDENCE: dict[str, float] = {
	"free_text_fallback": 0.75,
	"known_personnel_filename": 0.90,
	"anchored_identity": 0.97,
}


def _iter_json_files(path: Path):
	for file in sorted(path.glob("*.json")):
		if file.is_file():
			yield file


def _extract_name_candidates(value: str | None) -> list[str]:
	if not value:
		return []

	out: list[str] = []
	seen: set[str] = set()
	parts = ROLE_SEPARATOR_RE.split(value)
	for part in parts:
		token = part.strip()
		if not token:
			continue
		if DEPARTMENT_TOKEN_RE.match(token):
			continue

		canon = canonicalize_person_name(token)
		if not canon:
			continue
		if canon in seen:
			continue
		seen.add(canon)
		out.append(canon)

	return out


def _extract_names_from_free_text(value: str | None, allowed_ids: set[str] | None = None) -> list[str]:
	"""Extract probable person names from free text blocks (team members, author strings)."""
	if not value:
		return []

	out: list[str] = []
	seen: set[str] = set()
	for match in CAPITALIZED_NAME_RE.finditer(value):
		candidate = canonicalize_person_name(match.group(1))
		if not candidate:
			continue
		if len(candidate.split()) < 2:
			continue
		if allowed_ids is not None and candidate not in allowed_ids:
			continue
		if candidate in seen:
			continue
		seen.add(candidate)
		out.append(candidate)

	return out


def _extract_person_uid_from_url(value: str | None) -> str | None:
	if not isinstance(value, str):
		return None
	match = FIT_PERSON_UID_RE.search(value)
	return match.group(1) if match else None


def _load_identity_links(clean_root: Path) -> dict[str, str]:
	links_path = clean_root / "accumulated_personnel_links.json"
	if not links_path.is_file():
		return {}
	data = load_json(str(links_path))
	if not isinstance(data, dict):
		return {}
	return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _build_person_alias_map(people: dict[str, dict[str, Any]], identity_links: dict[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
	alias_to_person_id: dict[str, str] = {}
	anchor_by_person_id: dict[str, dict[str, str]] = {}

	for person_id, pdata in people.items():
		alias_to_person_id[person_id] = person_id
		pname = pdata.get("name")
		if isinstance(pname, str):
			pcanon = canonicalize_person_name(pname)
			if pcanon:
				alias_to_person_id[pcanon] = person_id

	for display_name, url in identity_links.items():
		alias = canonicalize_person_name(display_name)
		if not alias:
			continue
		person_id = alias_to_person_id.get(alias)
		if not person_id:
			continue

		alias_to_person_id[alias] = person_id
		anchor_by_person_id[person_id] = {
			"fit_profile_url": url,
			"fit_person_uid": _extract_person_uid_from_url(url) or "",
		}

	return alias_to_person_id, anchor_by_person_id


def _resolve_candidate_person_ids(candidates: list[str], alias_to_person_id: dict[str, str], known_person_ids: set[str]) -> list[str]:
	out: list[str] = []
	seen: set[str] = set()
	for candidate in candidates:
		canon = canonicalize_person_name(candidate)
		if not canon:
			continue
		resolved = alias_to_person_id.get(canon, canon)
		if resolved not in known_person_ids:
			continue
		if resolved in seen:
			continue
		seen.add(resolved)
		out.append(resolved)
	return out


def _edge_key(src_id: str, dst_id: str) -> str:
	return f"{src_id}|||{dst_id}"


def _edge_tuple_from_key(key: str) -> tuple[str, str]:
	left, right = key.split("|||", 1)
	return left, right


def _source_confidence(source: str) -> float:
	return EDGE_SOURCE_CONFIDENCE.get(source, 0.5)


def _source_priority(source: str) -> int:
	return EDGE_SOURCE_PRIORITY.get(source, 0)


def _choose_resolution_source(person_id: str, raw_candidate: str, alias_to_person_id: dict[str, str], anchored_person_ids: set[str], is_free_text: bool) -> str:
	canon = canonicalize_person_name(raw_candidate)
	alias_hit = bool(canon and canon in alias_to_person_id)
	if person_id in anchored_person_ids and alias_hit:
		return "anchored_identity"
	if is_free_text:
		return "free_text_fallback"
	return "known_personnel_filename"


def _resolve_candidate_person_links(
	candidates: list[str],
	alias_to_person_id: dict[str, str],
	known_person_ids: set[str],
	anchored_person_ids: set[str],
	is_free_text: bool,
) -> list[tuple[str, str]]:
	out: list[tuple[str, str]] = []
	seen: set[str] = set()
	for candidate in candidates:
		canon = canonicalize_person_name(candidate)
		if not canon:
			continue
		resolved = alias_to_person_id.get(canon, canon)
		if resolved not in known_person_ids:
			continue
		if resolved in seen:
			continue
		seen.add(resolved)
		out.append((resolved, _choose_resolution_source(resolved, candidate, alias_to_person_id, anchored_person_ids, is_free_text)))
	return out


def _merge_edge_evidence(
	evidence_by_rel: dict[str, dict[str, dict[str, Any]]],
	rel_type: str,
	src_id: str,
	dst_id: str,
	source: str,
	evidence_path: str,
	evidence_field: str,
) -> None:
	by_rel = evidence_by_rel.setdefault(rel_type, {})
	key = _edge_key(src_id, dst_id)
	candidate = {
		"source": source,
		"evidence_path": evidence_path,
		"evidence_field": evidence_field,
		"confidence": _source_confidence(source),
	}
	current = by_rel.get(key)
	if current is None:
		by_rel[key] = candidate
		return
	if candidate["confidence"] > current.get("confidence", 0):
		by_rel[key] = candidate
		return
	if candidate["confidence"] == current.get("confidence", 0):
		if _source_priority(candidate["source"]) > _source_priority(current.get("source", "")):
			by_rel[key] = candidate


def _extract_titles_from_list(items: Any) -> list[str]:
	if not isinstance(items, list):
		return []
	out = []
	for item in items:
		if isinstance(item, dict):
			title = item.get("title")
			if isinstance(title, str) and title.strip():
				out.append(title.strip())
	return out


def _extract_section_text(sections: dict[str, Any], key: str) -> str | None:
	obj = sections.get(key)
	if isinstance(obj, dict):
		text = obj.get("text")
		if isinstance(text, str) and text.strip():
			return text.strip()
	return None


def _split_multivalue_text(text: str | None) -> list[str]:
	if not text:
		return []
	parts = [p.strip() for p in MULTI_VALUE_SEP_RE.split(text) if p.strip()]
	return parts if parts else [text.strip()]


def _project_title_from_pub_value(value: str) -> str:
	# Publication project fields are often like: "Title , EU, PROGRAM, ...".
	return value.split(",", 1)[0].strip()


def _match_project_id_from_text(value: str | None, projects: dict[str, dict[str, Any]]) -> str | None:
	if not value:
		return None
	candidates = [value] + _split_multivalue_text(value)
	for candidate in candidates:
		title = _project_title_from_pub_value(candidate)
		cid = canonicalize_text(title)
		if cid in projects:
			return cid
		for pid, p in projects.items():
			ptitle = canonicalize_text(p.get("title") or "")
			if cid and (cid in ptitle or ptitle in cid):
				return pid
	return None


def _normalize_group_match_text(value: str) -> str:
	norm = canonicalize_text(value)
	norm = norm.replace(" rg ", " ")
	norm = re.sub(r"\s+", " ", norm).strip()
	return norm


def _match_group_id_from_text(value: str | None, groups: dict[str, dict[str, Any]]) -> str | None:
	if not value:
		return None
	candidates = [value] + _split_multivalue_text(value)
	for candidate in candidates:
		ccanon = _normalize_group_match_text(candidate)
		for gid, g in groups.items():
			gcanon = _normalize_group_match_text(g.get("group") or "")
			if ccanon == gcanon or (ccanon and (ccanon in gcanon or gcanon in ccanon)):
				return gid
	return None


def _department_id_from_text(value: str | None) -> tuple[str | None, str | None]:
	if not value:
		return None, None
	clean = value.strip()
	if not clean:
		return None, None
	return canonicalize_text(clean), clean


def _programme_id(programme_data: dict[str, Any], fallback_file_stem: str) -> str:
	metadata = programme_data.get("metadata") if isinstance(programme_data, dict) else None
	if isinstance(metadata, dict):
		abbr = metadata.get("abbreviation")
		if isinstance(abbr, str) and abbr.strip():
			return canonicalize_text(abbr)
	return canonicalize_text(fallback_file_stem)


def build_graph_snapshot(clean_root: Path) -> dict[str, Any]:
	persons_dir = clean_root / "personnel_profiles"
	courses_dir = clean_root / "courses"
	groups_dir = clean_root / "groups"
	projects_dir = clean_root / "projects"
	publications_dir = clean_root / "publications"

	people: dict[str, dict[str, Any]] = {}
	for path in _iter_json_files(persons_dir):
		data = load_json(str(path))
		rec = extract_person_contact_record(data, str(path))
		people[rec.profile_id] = {
			"profile_id": rec.profile_id,
			"name": rec.name,
			"name_normalized": rec.name_normalized,
			"email": rec.email,
			"office_room": rec.office_room,
			"has_email": rec.has_email,
			"has_office_room": rec.has_office_room,
			"source_path": rec.source_path,
		}

	known_person_ids = set(people.keys())
	identity_links = _load_identity_links(clean_root)
	alias_to_person_id, anchor_by_person_id = _build_person_alias_map(people, identity_links)
	anchored_person_ids = set(anchor_by_person_id.keys())
	for person_id, anchor in anchor_by_person_id.items():
		if person_id not in people:
			continue
		people[person_id]["fit_profile_url"] = anchor.get("fit_profile_url")
		people[person_id]["fit_person_uid"] = anchor.get("fit_person_uid")

	projects: dict[str, dict[str, Any]] = {}
	for path in _iter_json_files(projects_dir):
		data = load_json(str(path))
		title = data.get("title") if isinstance(data, dict) else None
		project_name = title if isinstance(title, str) and title.strip() else data.get("project")
		if not isinstance(project_name, str) or not project_name.strip():
			continue
		project_name = project_name.strip()
		project_id = canonicalize_text(project_name)

		metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
		sections = data.get("sections") if isinstance(data.get("sections"), dict) else {}
		team_text = None
		team_obj = sections.get("team_members") if isinstance(sections, dict) else None
		if isinstance(team_obj, dict):
			team_text = team_obj.get("text")

		projects[project_id] = {
			"project_id": project_id,
			"title": project_name,
			"code": metadata.get("code"),
			"agency": metadata.get("agency"),
			"program": metadata.get("program"),
			"project_period": metadata.get("project_period"),
			"team_member_links": _resolve_candidate_person_links(
				_extract_names_from_free_text(team_text if isinstance(team_text, str) else None),
				alias_to_person_id,
				known_person_ids,
				anchored_person_ids,
				is_free_text=True,
			),
			"source_path": str(path),
		}

	groups: dict[str, dict[str, Any]] = {}
	for path in _iter_json_files(groups_dir):
		data = load_json(str(path))
		group_name = data.get("group") if isinstance(data, dict) else None
		if not isinstance(group_name, str) or not group_name.strip():
			continue
		group_name = group_name.strip()
		group_id = canonicalize_text(group_name)

		team = data.get("team") if isinstance(data.get("team"), dict) else {}
		member_links: list[tuple[str, str]] = []
		member_seen: set[str] = set()
		for role_items in team.values() if isinstance(team, dict) else []:
			if not isinstance(role_items, list):
				continue
			for item in role_items:
				if not isinstance(item, dict):
					continue
				name = item.get("name")
				if not isinstance(name, str) or not name.strip():
					continue
				resolved_links = _resolve_candidate_person_links(
					[name],
					alias_to_person_id,
					known_person_ids,
					anchored_person_ids,
					is_free_text=False,
				)
				for person_id, source in resolved_links:
					if person_id in member_seen:
						continue
					member_seen.add(person_id)
					member_links.append((person_id, source))

		groups[group_id] = {
			"group_id": group_id,
			"group": group_name,
			"member_links": member_links,
			"project_titles": _extract_titles_from_list(data.get("projects")),
			"source_path": str(path),
		}

	publications: dict[str, dict[str, Any]] = {}
	departments: dict[str, dict[str, Any]] = {}
	for path in _iter_json_files(publications_dir):
		data = load_json(str(path))
		title = data.get("title") if isinstance(data, dict) else None
		if not isinstance(title, str) or not title.strip():
			continue
		title = title.strip()
		publication_id = canonicalize_text(title)

		sections = data.get("sections") if isinstance(data.get("sections"), dict) else {}
		authors_text = _extract_section_text(sections, "authors")
		projects_text = _extract_section_text(sections, "projects")
		groups_text = _extract_section_text(sections, "research_groups")
		departments_text = _extract_section_text(sections, "departments")

		project_id = _match_project_id_from_text(projects_text, projects)
		group_id = _match_group_id_from_text(groups_text, groups)
		department_id, department_name = _department_id_from_text(departments_text)
		if department_id and department_name and department_id not in departments:
			departments[department_id] = {
				"department_id": department_id,
				"name": department_name,
			}

		publications[publication_id] = {
			"publication_id": publication_id,
			"title": title,
			"year": data.get("year"),
			"author_links": _resolve_candidate_person_links(
				_extract_names_from_free_text(authors_text if isinstance(authors_text, str) else None),
				alias_to_person_id,
				known_person_ids,
				anchored_person_ids,
				is_free_text=True,
			),
			"projects_text": projects_text if isinstance(projects_text, str) else None,
			"research_groups_text": groups_text if isinstance(groups_text, str) else None,
			"departments_text": departments_text if isinstance(departments_text, str) else None,
			"project_id": project_id,
			"group_id": group_id,
			"department_id": department_id,
			"source_path": str(path),
		}

	courses: dict[str, dict[str, Any]] = {}
	teaches_guarantees: list[tuple[str, str, str]] = []
	relationship_evidence: dict[str, dict[str, dict[str, Any]]] = {}
	for path in _iter_json_files(courses_dir):
		data = load_json(str(path))
		rec = extract_course_semester_record(data, str(path))
		course_id = canonicalize_text(rec.code) if rec.code else rec.course_id

		content = data.get("content") if isinstance(data, dict) else {}
		content = content if isinstance(content, dict) else {}
		guarantor_links = _resolve_candidate_person_links(
			_extract_name_candidates(content.get("guarantor")),
			alias_to_person_id,
			known_person_ids,
			anchored_person_ids,
			is_free_text=False,
		)
		lecturer_links = _resolve_candidate_person_links(
			_extract_name_candidates(content.get("lecturer")),
			alias_to_person_id,
			known_person_ids,
			anchored_person_ids,
			is_free_text=False,
		)
		guarantor_names = [pid for pid, _source in guarantor_links]
		lecturer_names = [pid for pid, _source in lecturer_links]

		courses[course_id] = {
			"course_id": course_id,
			"code": rec.code,
			"course_name": rec.course_name,
			"semester_label": rec.semester_label,
			"semester_norm": rec.semester_norm,
			"has_semester": rec.has_semester,
			"semester_source_field": rec.semester_source_field,
			"source_path": rec.source_path,
			"guarantor_names": guarantor_names,
			"lecturer_names": lecturer_names,
		}

		for person_id, source in guarantor_links:
			teaches_guarantees.append((person_id, course_id, "GUARANTEES"))
			_merge_edge_evidence(
				relationship_evidence,
				"GUARANTEES",
				person_id,
				course_id,
				source,
				str(path),
				"content.guarantor",
			)
		for person_id, source in lecturer_links:
			teaches_guarantees.append((person_id, course_id, "LECTURES"))
			_merge_edge_evidence(
				relationship_evidence,
				"LECTURES",
				person_id,
				course_id,
				source,
				str(path),
				"content.lecturer",
			)

	programmes: dict[str, dict[str, Any]] = {}
	has_course: list[tuple[str, str]] = []
	for path in sorted(clean_root.glob("MIT*.json")):
		data = load_json(str(path))
		pid = _programme_id(data, path.stem)
		metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
		programmes[pid] = {
			"programme_id": pid,
			"abbreviation": metadata.get("abbreviation"),
			"code": metadata.get("code"),
			"language": metadata.get("language"),
			"source_path": str(path),
		}

		curriculum = data.get("curriculum", [])
		if not isinstance(curriculum, list):
			continue

		for sem in curriculum:
			if not isinstance(sem, dict):
				continue
			sem_label = sem.get("semester")
			for course in sem.get("courses", []):
				if not isinstance(course, dict):
					continue
				ccode = course.get("abbreviation")
				if not isinstance(ccode, str) or not ccode.strip():
					continue
				course_id = canonicalize_text(ccode)
				if course_id in courses:
					has_course.append((pid, course_id))

	member_of: list[tuple[str, str]] = []
	for group_id, g in groups.items():
		for person_id, source in g.get("member_links", []):
			member_of.append((person_id, group_id))
			_merge_edge_evidence(
				relationship_evidence,
				"MEMBER_OF",
				person_id,
				group_id,
				source,
				g["source_path"],
				"team.*[].name",
			)

	works_on: list[tuple[str, str]] = []
	# Person-side project links.
	for person_id, p in people.items():
		pdata = load_json(p["source_path"])
		for entry in pdata.get("projects", []) if isinstance(pdata.get("projects"), list) else []:
			if not isinstance(entry, dict):
				continue
			title = entry.get("title")
			if not isinstance(title, str) or not title.strip():
				continue
			project_id = canonicalize_text(title)
			if project_id in projects:
				works_on.append((person_id, project_id))
				_merge_edge_evidence(
					relationship_evidence,
					"WORKS_ON",
					person_id,
					project_id,
					"known_personnel_filename",
					p["source_path"],
					"projects[].title",
				)

	# Project-side team member links.
	for project_id, proj in projects.items():
		for person_id, source in proj.get("team_member_links", []):
			works_on.append((person_id, project_id))
			_merge_edge_evidence(
				relationship_evidence,
				"WORKS_ON",
				person_id,
				project_id,
				source,
				proj["source_path"],
				"sections.team_members",
			)

	authored: list[tuple[str, str]] = []
	for publication_id, pub in publications.items():
		for person_id, source in pub.get("author_links", []):
			authored.append((person_id, publication_id))
			_merge_edge_evidence(
				relationship_evidence,
				"AUTHORED",
				person_id,
				publication_id,
				source,
				pub["source_path"],
				"sections.authors",
			)

	runs_project: list[tuple[str, str]] = []
	for group_id, g in groups.items():
		for project_title in g.get("project_titles", []):
			project_id = canonicalize_text(project_title)
			if project_id in projects:
				runs_project.append((group_id, project_id))

	related_to_project: list[tuple[str, str]] = []
	related_to_group: list[tuple[str, str]] = []
	related_to_department: list[tuple[str, str]] = []
	for publication_id, pub in publications.items():
		pid = pub.get("project_id")
		gid = pub.get("group_id")
		did = pub.get("department_id")
		if isinstance(pid, str) and pid in projects:
			related_to_project.append((publication_id, pid))
		if isinstance(gid, str) and gid in groups:
			related_to_group.append((publication_id, gid))
		if isinstance(did, str) and did in departments:
			related_to_department.append((publication_id, did))

	# Deduplicate edges.
	def _dedupe_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
		return sorted(set(edges))

	rels: dict[str, list[tuple[str, str]]] = {
		"HAS_COURSE": has_course,
		"GUARANTEES": [(p, c) for p, c, r in teaches_guarantees if r == "GUARANTEES"],
		"LECTURES": [(p, c) for p, c, r in teaches_guarantees if r == "LECTURES"],
		"MEMBER_OF": _dedupe_edges(member_of),
		"WORKS_ON": _dedupe_edges(works_on),
		"AUTHORED": _dedupe_edges(authored),
		"RUNS_PROJECT": _dedupe_edges(runs_project),
		"RELATED_TO_PROJECT": _dedupe_edges(related_to_project),
		"RELATED_TO_GROUP": _dedupe_edges(related_to_group),
		"RELATED_TO_DEPARTMENT": _dedupe_edges(related_to_department),
	}

	return {
		"people": people,
		"courses": courses,
		"groups": groups,
		"projects": projects,
		"publications": publications,
		"departments": departments,
		"programmes": programmes,
		"relationships": rels,
		"relationship_evidence": {
			rel: {k: v for k, v in by_edge.items()} for rel, by_edge in relationship_evidence.items()
		},
	}


def _resolve_person_id(snapshot: dict[str, Any], person_query: str) -> tuple[str | None, list[str]]:
	q = canonicalize_person_name(person_query)
	if not q:
		return None, []
	if q in snapshot["people"]:
		return q, []

	candidates = [pid for pid in snapshot["people"] if q in pid or pid in q]
	candidates = sorted(set(candidates))[:5]
	return (candidates[0], candidates[1:]) if candidates else (None, [])


def _resolve_course_id(snapshot: dict[str, Any], course_query: str) -> tuple[str | None, list[str]]:
	q = canonicalize_text(course_query)
	if not q:
		return None, []

	# Match by code first
	for cid, data in snapshot["courses"].items():
		code = data.get("code")
		if isinstance(code, str) and canonicalize_text(code) == q:
			return cid, []

	# Fallback to title contains
	candidates = []
	for cid, data in snapshot["courses"].items():
		title = data.get("course_name")
		tcanon = canonicalize_text(title) if isinstance(title, str) else ""
		if q in tcanon or tcanon in q:
			candidates.append(cid)

	candidates = sorted(set(candidates))[:5]
	return (candidates[0], candidates[1:]) if candidates else (None, [])


def query_person_contact(snapshot: dict[str, Any], person_query: str) -> dict[str, Any]:
	pid, alternatives = _resolve_person_id(snapshot, person_query)
	if not pid:
		return {"query": person_query, "found": False, "message": "person not found", "candidates": []}

	p = snapshot["people"][pid]
	return {
		"query": person_query,
		"found": True,
		"person_id": pid,
		"person_name": p.get("name") or pid,
		"email": p.get("email"),
		"office_room": p.get("office_room"),
		"has_email": p.get("has_email"),
		"has_office_room": p.get("has_office_room"),
		"other_candidates": alternatives,
	}


def query_course_semester(snapshot: dict[str, Any], course_query: str) -> dict[str, Any]:
	cid, alternatives = _resolve_course_id(snapshot, course_query)
	if not cid:
		return {"query": course_query, "found": False, "message": "course not found", "candidates": []}

	c = snapshot["courses"][cid]
	return {
		"query": course_query,
		"found": True,
		"course_id": cid,
		"course_code": c.get("code"),
		"course_name": c.get("course_name"),
		"semester_label": c.get("semester_label"),
		"semester_norm": c.get("semester_norm"),
		"other_candidates": alternatives,
	}


def query_course_staff(snapshot: dict[str, Any], course_query: str) -> dict[str, Any]:
	cid, alternatives = _resolve_course_id(snapshot, course_query)
	if not cid:
		return {"query": course_query, "found": False, "message": "course not found", "candidates": []}

	c = snapshot["courses"][cid]
	people = snapshot["people"]

	def _render_people(ids: list[str]) -> list[dict[str, Any]]:
		out = []
		for pid in ids:
			p = people.get(pid)
			if p:
				out.append(
					{
						"person_id": pid,
						"name": p.get("name") or pid,
						"email": p.get("email"),
						"office_room": p.get("office_room"),
					}
				)
			else:
				out.append({"person_id": pid, "name": pid, "email": None, "office_room": None})
		return out

	return {
		"query": course_query,
		"found": True,
		"course_code": c.get("code"),
		"course_name": c.get("course_name"),
		"guarantors": _render_people(c.get("guarantor_names", [])),
		"lecturers": _render_people(c.get("lecturer_names", [])),
		"other_candidates": alternatives,
	}


def query_programme_courses(snapshot: dict[str, Any], programme_query: str, semester_norm: str | None = None) -> dict[str, Any]:
	q = canonicalize_text(programme_query)
	pid = None
	for program_id, p in snapshot["programmes"].items():
		abbr = p.get("abbreviation")
		if isinstance(abbr, str) and canonicalize_text(abbr) == q:
			pid = program_id
			break
		if q in program_id:
			pid = program_id
			break

	if not pid:
		return {"query": programme_query, "found": False, "message": "programme not found"}

	linked = [c for prog, c in snapshot["relationships"]["HAS_COURSE"] if prog == pid]
	courses = []
	for cid in linked:
		c = snapshot["courses"].get(cid)
		if not c:
			continue
		if semester_norm and c.get("semester_norm") != semester_norm:
			continue
		courses.append({"code": c.get("code"), "course_name": c.get("course_name"), "semester_norm": c.get("semester_norm")})

	courses = sorted(courses, key=lambda x: (x.get("code") or ""))
	return {
		"query": programme_query,
		"found": True,
		"programme_id": pid,
		"programme_abbreviation": snapshot["programmes"][pid].get("abbreviation"),
		"semester_filter": semester_norm,
		"course_count": len(courses),
		"courses": courses,
	}


def get_template_outputs(snapshot: dict[str, Any]) -> dict[str, Any]:
	# These are deterministic demo templates for validation/debugging.
	member_of_edges = snapshot["relationships"].get("MEMBER_OF", [])
	works_on_edges = snapshot["relationships"].get("WORKS_ON", [])
	authored_edges = snapshot["relationships"].get("AUTHORED", [])
	rel_pub_proj = snapshot["relationships"].get("RELATED_TO_PROJECT", [])
	rel_pub_group = snapshot["relationships"].get("RELATED_TO_GROUP", [])
	rel_pub_dept = snapshot["relationships"].get("RELATED_TO_DEPARTMENT", [])
	return {
		"person_contact_example": query_person_contact(snapshot, "chudy peter"),
		"person_contact_existing_example": query_person_contact(snapshot, "barina david"),
		"course_semester_example": query_course_semester(snapshot, "BAYa"),
		"course_staff_example": query_course_staff(snapshot, "BAYa"),
		"programme_courses_winter_example": query_programme_courses(snapshot, "MIT-EN", semester_norm="winter"),
		"member_of_example": member_of_edges[0] if member_of_edges else None,
		"works_on_example": works_on_edges[0] if works_on_edges else None,
		"authored_example": authored_edges[0] if authored_edges else None,
		"publication_related_project_example": rel_pub_proj[0] if rel_pub_proj else None,
		"publication_related_group_example": rel_pub_group[0] if rel_pub_group else None,
		"publication_related_department_example": rel_pub_dept[0] if rel_pub_dept else None,
	}


def get_cypher_validation_templates() -> dict[str, str]:
	"""Cypher templates used to sanity-check key relation families in Neo4j."""
	return {
		"person_contact": """
			MATCH (p:Person {profile_id: $person_id})
			RETURN p.profile_id AS person_id, p.name AS name, p.email AS email, p.office_room AS office_room
		""",
		"course_semester": """
			MATCH (c:Course {code: $course_code})
			RETURN c.code AS course_code, c.course_name AS course_name, c.semester_label AS semester_label, c.semester_norm AS semester_norm
		""",
		"publication_related_project": """
			MATCH (pub:Publication {publication_id: $publication_id})-[:RELATED_TO_PROJECT]->(pr:Project)
			RETURN pub.title AS publication_title, pr.title AS project_title
		""",
		"publication_related_group": """
			MATCH (pub:Publication {publication_id: $publication_id})-[:RELATED_TO_GROUP]->(g:ResearchGroup)
			RETURN pub.title AS publication_title, g.group AS group_name
		""",
		"publication_related_department": """
			MATCH (pub:Publication {publication_id: $publication_id})-[:RELATED_TO_DEPARTMENT]->(d:Department)
			RETURN pub.title AS publication_title, d.name AS department_name
		""",
	}


def run_db_template_validation(uri: str, user: str, password: str, database: str) -> dict[str, Any]:
	try:
		from neo4j import GraphDatabase  # type: ignore
	except ImportError as exc:
		raise RuntimeError("neo4j Python driver is not installed; cannot run DB template validation.") from exc

	templates = get_cypher_validation_templates()
	params = {
		"person_contact": {"person_id": "barina david"},
		"course_semester": {"course_code": "BAYa"},
		"publication_related_project": {"publication_id": "focus-aware compression and image quality metric for 3d displays"},
		"publication_related_group": {"publication_id": "focus-aware compression and image quality metric for 3d displays"},
		"publication_related_department": {"publication_id": "focus-aware compression and image quality metric for 3d displays"},
	}

	driver = GraphDatabase.driver(uri, auth=(user, password))
	results: dict[str, Any] = {}
	with driver.session(database=database) as session:
		for name, query in templates.items():
			records = session.run(query, **params.get(name, {})).data()
			results[name] = {
				"row_count": len(records),
				"sample": records[0] if records else None,
			}
	driver.close()
	return results


def _apply_to_neo4j(snapshot: dict[str, Any], uri: str, user: str, password: str, database: str) -> dict[str, int]:
	try:
		from neo4j import GraphDatabase  # type: ignore
	except ImportError as exc:
		raise RuntimeError("neo4j Python driver is not installed; run in --dry_run mode or install neo4j package.") from exc

	driver = GraphDatabase.driver(uri, auth=(user, password))
	rel_evidence: dict[str, dict[str, dict[str, Any]]] = snapshot.get("relationship_evidence", {})

	def _edge_evidence(rel_type: str, src_id: str, dst_id: str, default_source: str, default_path: str, default_field: str) -> dict[str, Any]:
		default = {
			"source": default_source,
			"evidence_path": default_path,
			"evidence_field": default_field,
			"confidence": _source_confidence(default_source),
		}
		by_rel = rel_evidence.get(rel_type, {})
		return by_rel.get(_edge_key(src_id, dst_id), default)

	person_rows = list(snapshot["people"].values())
	course_rows = list(snapshot["courses"].values())
	group_rows = list(snapshot.get("groups", {}).values())
	project_rows = list(snapshot.get("projects", {}).values())
	publication_rows = list(snapshot.get("publications", {}).values())
	department_rows = list(snapshot.get("departments", {}).values())
	programme_rows = list(snapshot["programmes"].values())

	with driver.session(database=database) as session:
		session.run("CREATE CONSTRAINT person_profile_id IF NOT EXISTS FOR (p:Person) REQUIRE p.profile_id IS UNIQUE")
		session.run("CREATE CONSTRAINT course_code IF NOT EXISTS FOR (c:Course) REQUIRE c.code IS UNIQUE")
		session.run("CREATE CONSTRAINT group_id IF NOT EXISTS FOR (g:ResearchGroup) REQUIRE g.group_id IS UNIQUE")
		session.run("CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE")
		session.run("CREATE CONSTRAINT publication_id IF NOT EXISTS FOR (p:Publication) REQUIRE p.publication_id IS UNIQUE")
		session.run("CREATE CONSTRAINT department_id IF NOT EXISTS FOR (d:Department) REQUIRE d.department_id IS UNIQUE")
		session.run("CREATE CONSTRAINT programme_id IF NOT EXISTS FOR (p:Programme) REQUIRE p.programme_id IS UNIQUE")

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (p:Person {profile_id: row.profile_id})
			SET p.name = row.name,
				p.name_normalized = row.name_normalized,
				p.email = row.email,
				p.office_room = row.office_room,
				p.fit_profile_url = row.fit_profile_url,
				p.fit_person_uid = row.fit_person_uid,
				p.has_email = row.has_email,
				p.has_office_room = row.has_office_room,
				p.source_path = row.source_path
			""",
			rows=person_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (c:Course {code: row.code})
			SET c.course_id = row.course_id,
				c.course_name = row.course_name,
				c.semester_label = row.semester_label,
				c.semester_norm = row.semester_norm,
				c.has_semester = row.has_semester,
				c.semester_source_field = row.semester_source_field,
				c.source_path = row.source_path
			""",
			rows=[r for r in course_rows if r.get("code")],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (g:ResearchGroup {group_id: row.group_id})
			SET g.group = row.group,
				g.source_path = row.source_path
			""",
			rows=group_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (p:Project {project_id: row.project_id})
			SET p.title = row.title,
				p.code = row.code,
				p.agency = row.agency,
				p.program = row.program,
				p.project_period = row.project_period,
				p.source_path = row.source_path
			""",
			rows=project_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (p:Publication {publication_id: row.publication_id})
			SET p.title = row.title,
				p.year = row.year,
				p.source_path = row.source_path
			""",
			rows=publication_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (d:Department {department_id: row.department_id})
			SET d.name = row.name
			""",
			rows=department_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MERGE (p:Programme {programme_id: row.programme_id})
			SET p.abbreviation = row.abbreviation,
				p.code = row.code,
				p.language = row.language,
				p.source_path = row.source_path
			""",
			rows=programme_rows,
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Programme {programme_id: row.programme_id})
			MATCH (c:Course {code: row.course_code})
			MERGE (p)-[:HAS_COURSE]->(c)
			""",
			rows=[{"programme_id": p, "course_code": snapshot["courses"][c]["code"]} for p, c in snapshot["relationships"]["HAS_COURSE"] if snapshot["courses"][c].get("code")],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Person {profile_id: row.person_id})
			MATCH (c:Course {code: row.course_code})
			MERGE (p)-[r:GUARANTEES]->(c)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"person_id": p,
					"course_code": snapshot["courses"][c]["code"],
					"source": _edge_evidence("GUARANTEES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.guarantor")["source"],
					"evidence_path": _edge_evidence("GUARANTEES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.guarantor")["evidence_path"],
					"evidence_field": _edge_evidence("GUARANTEES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.guarantor")["evidence_field"],
					"confidence": _edge_evidence("GUARANTEES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.guarantor")["confidence"],
				}
				for p, c in snapshot["relationships"]["GUARANTEES"]
				if p in snapshot["people"] and snapshot["courses"][c].get("code")
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Person {profile_id: row.person_id})
			MATCH (c:Course {code: row.course_code})
			MERGE (p)-[r:LECTURES]->(c)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"person_id": p,
					"course_code": snapshot["courses"][c]["code"],
					"source": _edge_evidence("LECTURES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.lecturer")["source"],
					"evidence_path": _edge_evidence("LECTURES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.lecturer")["evidence_path"],
					"evidence_field": _edge_evidence("LECTURES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.lecturer")["evidence_field"],
					"confidence": _edge_evidence("LECTURES", p, c, "known_personnel_filename", snapshot["courses"][c]["source_path"], "content.lecturer")["confidence"],
				}
				for p, c in snapshot["relationships"]["LECTURES"]
				if p in snapshot["people"] and snapshot["courses"][c].get("code")
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Person {profile_id: row.person_id})
			MATCH (g:ResearchGroup {group_id: row.group_id})
			MERGE (p)-[r:MEMBER_OF]->(g)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"person_id": p,
					"group_id": g,
					"source": _edge_evidence("MEMBER_OF", p, g, "known_personnel_filename", snapshot["groups"][g]["source_path"], "team.*[].name")["source"],
					"evidence_path": _edge_evidence("MEMBER_OF", p, g, "known_personnel_filename", snapshot["groups"][g]["source_path"], "team.*[].name")["evidence_path"],
					"evidence_field": _edge_evidence("MEMBER_OF", p, g, "known_personnel_filename", snapshot["groups"][g]["source_path"], "team.*[].name")["evidence_field"],
					"confidence": _edge_evidence("MEMBER_OF", p, g, "known_personnel_filename", snapshot["groups"][g]["source_path"], "team.*[].name")["confidence"],
				}
				for p, g in snapshot["relationships"].get("MEMBER_OF", [])
				if p in snapshot["people"] and g in snapshot["groups"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Person {profile_id: row.person_id})
			MATCH (pr:Project {project_id: row.project_id})
			MERGE (p)-[r:WORKS_ON]->(pr)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"person_id": p,
					"project_id": pr,
					"source": _edge_evidence("WORKS_ON", p, pr, "free_text_fallback", snapshot["projects"][pr]["source_path"], "sections.team_members|personnel.projects")["source"],
					"evidence_path": _edge_evidence("WORKS_ON", p, pr, "free_text_fallback", snapshot["projects"][pr]["source_path"], "sections.team_members|personnel.projects")["evidence_path"],
					"evidence_field": _edge_evidence("WORKS_ON", p, pr, "free_text_fallback", snapshot["projects"][pr]["source_path"], "sections.team_members|personnel.projects")["evidence_field"],
					"confidence": _edge_evidence("WORKS_ON", p, pr, "free_text_fallback", snapshot["projects"][pr]["source_path"], "sections.team_members|personnel.projects")["confidence"],
				}
				for p, pr in snapshot["relationships"].get("WORKS_ON", [])
				if p in snapshot["people"] and pr in snapshot["projects"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (p:Person {profile_id: row.person_id})
			MATCH (pub:Publication {publication_id: row.publication_id})
			MERGE (p)-[r:AUTHORED]->(pub)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"person_id": p,
					"publication_id": pub,
					"source": _edge_evidence("AUTHORED", p, pub, "free_text_fallback", snapshot["publications"][pub]["source_path"], "sections.authors")["source"],
					"evidence_path": _edge_evidence("AUTHORED", p, pub, "free_text_fallback", snapshot["publications"][pub]["source_path"], "sections.authors")["evidence_path"],
					"evidence_field": _edge_evidence("AUTHORED", p, pub, "free_text_fallback", snapshot["publications"][pub]["source_path"], "sections.authors")["evidence_field"],
					"confidence": _edge_evidence("AUTHORED", p, pub, "free_text_fallback", snapshot["publications"][pub]["source_path"], "sections.authors")["confidence"],
				}
				for p, pub in snapshot["relationships"].get("AUTHORED", [])
				if p in snapshot["people"] and pub in snapshot["publications"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (g:ResearchGroup {group_id: row.group_id})
			MATCH (pr:Project {project_id: row.project_id})
			MERGE (g)-[r:RUNS_PROJECT]->(pr)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"group_id": g,
					"project_id": pr,
					"source": "cleaned",
					"evidence_path": snapshot["groups"][g]["source_path"],
					"evidence_field": "projects[].title",
					"confidence": 0.85,
				}
				for g, pr in snapshot["relationships"].get("RUNS_PROJECT", [])
				if g in snapshot["groups"] and pr in snapshot["projects"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (pub:Publication {publication_id: row.publication_id})
			MATCH (pr:Project {project_id: row.project_id})
			MERGE (pub)-[r:RELATED_TO_PROJECT]->(pr)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"publication_id": pub,
					"project_id": pr,
					"source": "cleaned",
					"evidence_path": snapshot["publications"][pub]["source_path"],
					"evidence_field": "sections.projects",
					"confidence": 0.8,
				}
				for pub, pr in snapshot["relationships"].get("RELATED_TO_PROJECT", [])
				if pub in snapshot["publications"] and pr in snapshot["projects"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (pub:Publication {publication_id: row.publication_id})
			MATCH (g:ResearchGroup {group_id: row.group_id})
			MERGE (pub)-[r:RELATED_TO_GROUP]->(g)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"publication_id": pub,
					"group_id": g,
					"source": "cleaned",
					"evidence_path": snapshot["publications"][pub]["source_path"],
					"evidence_field": "sections.research_groups",
					"confidence": 0.8,
				}
				for pub, g in snapshot["relationships"].get("RELATED_TO_GROUP", [])
				if pub in snapshot["publications"] and g in snapshot["groups"]
			],
		)

		session.run(
			"""
			UNWIND $rows AS row
			MATCH (pub:Publication {publication_id: row.publication_id})
			MATCH (d:Department {department_id: row.department_id})
			MERGE (pub)-[r:RELATED_TO_DEPARTMENT]->(d)
			SET r.source = row.source,
				r.evidence_path = row.evidence_path,
				r.evidence_field = row.evidence_field,
				r.confidence = row.confidence
			""",
			rows=[
				{
					"publication_id": pub,
					"department_id": d,
					"source": "cleaned",
					"evidence_path": snapshot["publications"][pub]["source_path"],
					"evidence_field": "sections.departments",
					"confidence": 0.8,
				}
				for pub, d in snapshot["relationships"].get("RELATED_TO_DEPARTMENT", [])
				if pub in snapshot["publications"] and d in snapshot.get("departments", {})
			],
		)

	driver.close()

	return {
		"people": len(person_rows),
		"courses": len([r for r in course_rows if r.get("code")]),
		"groups": len(group_rows),
		"projects": len(project_rows),
		"publications": len(publication_rows),
		"departments": len(department_rows),
		"programmes": len(programme_rows),
		"has_course_edges": len(snapshot["relationships"]["HAS_COURSE"]),
		"guarantees_edges": len(snapshot["relationships"]["GUARANTEES"]),
		"lectures_edges": len(snapshot["relationships"]["LECTURES"]),
		"member_of_edges": len(snapshot["relationships"].get("MEMBER_OF", [])),
		"works_on_edges": len(snapshot["relationships"].get("WORKS_ON", [])),
		"authored_edges": len(snapshot["relationships"].get("AUTHORED", [])),
		"runs_project_edges": len(snapshot["relationships"].get("RUNS_PROJECT", [])),
		"related_to_project_edges": len(snapshot["relationships"].get("RELATED_TO_PROJECT", [])),
		"related_to_group_edges": len(snapshot["relationships"].get("RELATED_TO_GROUP", [])),
		"related_to_department_edges": len(snapshot["relationships"].get("RELATED_TO_DEPARTMENT", [])),
	}


def _snapshot_stats(snapshot: dict[str, Any]) -> dict[str, Any]:
	rel_counts = {k: len(v) for k, v in snapshot["relationships"].items()}
	unresolved_by_rel = defaultdict(int)
	for rel_type in ["GUARANTEES", "LECTURES"]:
		for person_id, _course_id in snapshot["relationships"][rel_type]:
			if person_id not in snapshot["people"]:
				unresolved_by_rel[rel_type] += 1

	return {
		"people": len(snapshot["people"]),
		"courses": len(snapshot["courses"]),
		"groups": len(snapshot.get("groups", {})),
		"projects": len(snapshot.get("projects", {})),
		"publications": len(snapshot.get("publications", {})),
		"departments": len(snapshot.get("departments", {})),
		"programmes": len(snapshot["programmes"]),
		"relationships": rel_counts,
		"unresolved_staff_edges": dict(unresolved_by_rel),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="MVP graph builder and optional Neo4j ingester for FIT cleaned data")
	parser.add_argument("--clean_root", type=str, default="extracted_data_clean/fit")
	parser.add_argument("--dry_run", action="store_true", help="Build snapshot and print stats without writing to Neo4j")
	parser.add_argument("--print_templates", action="store_true", help="Print sample template outputs for entity relationship QA")
	parser.add_argument("--neo4j_uri", type=str, default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
	parser.add_argument("--neo4j_user", type=str, default=os.environ.get("NEO4J_USER", "neo4j"))
	parser.add_argument("--neo4j_password", type=str, default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
	parser.add_argument("--neo4j_database", type=str, default=os.environ.get("NEO4J_DATABASE", "neo4j"))
	parser.add_argument("--validate_db_templates", action="store_true", help="Run DB-backed Cypher template validations after write")
	args = parser.parse_args()

	clean_root = Path(args.clean_root).resolve()
	snapshot = build_graph_snapshot(clean_root)

	print("[snapshot_stats]")
	print(json.dumps(_snapshot_stats(snapshot), indent=2, ensure_ascii=False))

	if args.print_templates:
		print("[template_outputs]")
		print(json.dumps(get_template_outputs(snapshot), indent=2, ensure_ascii=False))

	if not args.dry_run:
		result = _apply_to_neo4j(
			snapshot=snapshot,
			uri=args.neo4j_uri,
			user=args.neo4j_user,
			password=args.neo4j_password,
			database=args.neo4j_database,
		)
		print("[neo4j_apply]")
		print(json.dumps(result, indent=2, ensure_ascii=False))

		if args.validate_db_templates:
			template_results = run_db_template_validation(
				uri=args.neo4j_uri,
				user=args.neo4j_user,
				password=args.neo4j_password,
				database=args.neo4j_database,
			)
			print("[db_template_validation]")
			print(json.dumps(template_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
