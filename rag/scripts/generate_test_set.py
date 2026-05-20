import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


POINT_PATTERNS = {
    "midterm": [
        re.compile(r"(\d+)\s*pts?[^\n]*mid[- ]?term", re.IGNORECASE),
        re.compile(r"mid[- ]?term[^\n]*?(\d+)\s*points", re.IGNORECASE),
        re.compile(r"half[- ]?semestral[^\n]*?(\d+)\s*pts?", re.IGNORECASE),
    ],
    "final": [
        re.compile(r"(\d+)\s*pts?[^\n]*final exam", re.IGNORECASE),
        re.compile(r"final exam[^\n]*?(\d+)\s*points", re.IGNORECASE),
        re.compile(r"semestral exam[^\n]*?(\d+)\s*pts?", re.IGNORECASE),
    ],
    "project": [
        re.compile(r"(\d+)\s*pts?[^\n]*projects?", re.IGNORECASE),
        re.compile(r"projects?[^\n]*?(\d+)\s*points", re.IGNORECASE),
        re.compile(r"presentation of project[^\n]*?(\d+)\s*points", re.IGNORECASE),
    ],
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def parse_people(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|")]
    names: list[str] = []

    for p in parts:
        if not p:
            continue
        # Department tags like (DCGM)
        if p.startswith("(") and p.endswith(")"):
            continue
        if p.startswith("("):
            continue
        names.append(clean_text(p))

    deduped = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def parse_department(raw: str) -> str | None:
    if not raw:
        return None
    part = raw.split("|")[0]
    return clean_text(part) if clean_text(part) else None


def normalise_credits(raw: str | None) -> str | None:
    if not raw:
        return None
    text = clean_text(raw)
    m = re.search(r"(\d+)\s*credits?", text, re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1)} credits"


def normalise_semester(raw: str | None) -> str | None:
    if not raw:
        return None
    text = clean_text(raw)
    if re.search(r"\b(winter|summer)\b", text, re.IGNORECASE):
        return text
    return None


def extract_points(text: str, point_type: str) -> str | None:
    if not text:
        return None
    for pat in POINT_PATTERNS[point_type]:
        m = pat.search(text)
        if m:
            return f"{m.group(1)} pts"
    return None


def parse_first_textbook(content: dict) -> str | None:
    for key in ("fundamental_literature", "study_literature"):
        raw = content.get(key)
        if not raw:
            continue
        lines = [clean_text(x) for x in raw.split("\n") if clean_text(x)]
        # keep first non-url bullet
        for line in lines:
            line = re.sub(r"^-\s*", "", line)
            if line.lower().startswith("http"):
                continue
            line = re.sub(r"\s*isbn.*$", "", line, flags=re.IGNORECASE)
            line = clean_text(line)
            if line:
                return line[:180]
    return None


def has_project_component(content: dict) -> bool:
    blobs = [
        content.get("time_span", ""),
        content.get("assessment_points", ""),
        content.get("progress_assessment", ""),
        content.get("syllabus_-_others,_projects_and_individual_work_of_students", ""),
    ]
    text = "\n".join([b for b in blobs if b])
    return bool(re.search(r"\bproject(s)?\b", text, re.IGNORECASE))


def is_summer(semester: str | None) -> bool:
    return bool(semester and "summer" in semester.lower())


def course_record(course_path: Path) -> dict:
    data = load_json(course_path)
    metadata = data.get("metadata", {})
    content = data.get("content", {})

    code = metadata.get("code") or course_path.stem
    lecturer_names = parse_people(content.get("lecturer", ""))
    instructor_names = parse_people(content.get("instructor", ""))
    teachers = []
    seen = set()
    for name in lecturer_names + instructor_names:
        if name not in seen:
            seen.add(name)
            teachers.append(name)

    assessment_blob = "\n".join(
        [content.get("assessment_points", ""), content.get("progress_assessment", "")]
    )

    return {
        "code": code,
        "course_name": clean_text(metadata.get("course_name", "")) or None,
        "credits": normalise_credits(metadata.get("credits")),
        "semester": normalise_semester(metadata.get("semester")),
        "department": parse_department(content.get("department", "")),
        "teachers": teachers,
        "midterm_points": extract_points(assessment_blob, "midterm"),
        "final_points": extract_points(assessment_blob, "final"),
        "project_points": extract_points(assessment_blob, "project"),
        "has_project": has_project_component(content),
        "textbook": parse_first_textbook(content),
    }


def pick_template(rng: random.Random, templates: list[str], **kwargs) -> str:
    return rng.choice(templates).format(**kwargs)


def generate_candidate_questions(courses: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates: list[dict] = []

    all_teachers = sorted({t for c in courses for t in c["teachers"]})

    q_templates = {
        "midterm": [
            "How many points is the mid-term worth in {code}?",
            "Mid-term points for {code}?",
            "How many points for midterm in {code}?",
        ],
        "final": [
            "How many points is the final exam in {code}?",
            "Final exam points for {code}?",
            "How many points for final exam in {code}?",
        ],
        "project_points": [
            "How many points are for project work in {code}?",
            "Project points in {code}?",
            "How many points for project in {code}?",
        ],
        "has_project": [
            "Is there a project component in {code}?",
            "Does {code} include a project?",
            "Is project work required in {code}?",
        ],
        "who_teaches": [
            "Who teaches {code}?",
            "Who is teaching {code}?",
            "Which teachers are assigned to {code}?",
        ],
        "is_teacher": [
            "Is {teacher} teaching {code}?",
            "Does {teacher} teach {code}?",
            "Is {teacher} one of the teachers of {code}?",
        ],
        "summer": [
            "Is {code} taught in summer semester?",
            "Is {code} offered in the summer semester?",
            "Does {code} run in summer semester?",
        ],
        "credits": [
            "How many credits is {code}?",
            "How many credits for {code} course?",
            "What is the credit value of {code}?",
        ],
        "textbook": [
            "Which textbook is useful for {code}?",
            "Recommend one textbook for {code}.",
            "What is one key textbook for {code}?",
        ],
        "department": [
            "Which department is responsible for {code}?",
            "What department teaches {code}?",
            "Which department runs {code}?",
        ],
        "stands_for": [
            "What does {code} stand for?",
            "{code} stands for what course name?",
            "What is the full name of {code}?",
        ],
        "course_code_for_name": [
            "What is the course code for {course_name}?",
            "Which code corresponds to {course_name}?",
            "What code is used for {course_name}?",
        ],
    }

    for c in courses:
        code = c["code"]

        if c["midterm_points"]:
            candidates.append({
                "question": pick_template(rng, q_templates["midterm"], code=code),
                "expected_answer": c["midterm_points"],
                "answer_field": "content.assessment_points/progress_assessment(mid-term)",
                "course_code": code,
                "type": "midterm_points",
            })

        if c["final_points"]:
            candidates.append({
                "question": pick_template(rng, q_templates["final"], code=code),
                "expected_answer": c["final_points"],
                "answer_field": "content.assessment_points/progress_assessment(final)",
                "course_code": code,
                "type": "final_points",
            })

        if c["project_points"]:
            candidates.append({
                "question": pick_template(rng, q_templates["project_points"], code=code),
                "expected_answer": c["project_points"],
                "answer_field": "content.assessment_points/progress_assessment(project)",
                "course_code": code,
                "type": "project_points",
            })

        candidates.append({
            "question": pick_template(rng, q_templates["has_project"], code=code),
            "expected_answer": "Yes" if c["has_project"] else "No",
            "answer_field": "content.time_span/assessment/progress(project presence)",
            "course_code": code,
            "type": "has_project",
        })

        if c["teachers"]:
            short_teacher_list = ", ".join(c["teachers"][:4])
            candidates.append({
                "question": pick_template(rng, q_templates["who_teaches"], code=code),
                "expected_answer": short_teacher_list,
                "answer_field": "content.lecturer/content.instructor",
                "course_code": code,
                "type": "who_teaches",
            })

            true_teacher = rng.choice(c["teachers"])
            candidates.append({
                "question": pick_template(rng, q_templates["is_teacher"], teacher=true_teacher, code=code),
                "expected_answer": "Yes",
                "answer_field": "content.lecturer/content.instructor",
                "course_code": code,
                "type": "is_teacher_yes",
            })

            negatives = [t for t in all_teachers if t not in c["teachers"]]
            if negatives:
                false_teacher = rng.choice(negatives)
                candidates.append({
                    "question": pick_template(rng, q_templates["is_teacher"], teacher=false_teacher, code=code),
                    "expected_answer": "No",
                    "answer_field": "content.lecturer/content.instructor",
                    "course_code": code,
                    "type": "is_teacher_no",
                })

        if c["semester"]:
            candidates.append({
                "question": pick_template(rng, q_templates["summer"], code=code),
                "expected_answer": "Yes" if is_summer(c["semester"]) else "No",
                "answer_field": "metadata.semester",
                "course_code": code,
                "type": "is_summer",
            })

        if c["credits"]:
            candidates.append({
                "question": pick_template(rng, q_templates["credits"], code=code),
                "expected_answer": c["credits"],
                "answer_field": "metadata.credits",
                "course_code": code,
                "type": "credits",
            })

        if c["textbook"]:
            candidates.append({
                "question": pick_template(rng, q_templates["textbook"], code=code),
                "expected_answer": c["textbook"],
                "answer_field": "content.fundamental_literature/content.study_literature(first item)",
                "course_code": code,
                "type": "textbook",
            })

        if c["department"]:
            candidates.append({
                "question": pick_template(rng, q_templates["department"], code=code),
                "expected_answer": c["department"],
                "answer_field": "content.department",
                "course_code": code,
                "type": "department",
            })

        if c["course_name"]:
            candidates.append({
                "question": pick_template(rng, q_templates["stands_for"], code=code),
                "expected_answer": c["course_name"],
                "answer_field": "metadata.course_name",
                "course_code": code,
                "type": "code_expansion",
            })

            candidates.append({
                "question": pick_template(rng, q_templates["course_code_for_name"], course_name=c["course_name"]),
                "expected_answer": code,
                "answer_field": "metadata.code",
                "course_code": code,
                "type": "code_lookup",
            })

    # Deduplicate exact question strings.
    unique = {}
    for item in candidates:
        unique[item["question"]] = item
    return list(unique.values())


def sample_questions(candidates: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(candidates) <= n:
        sampled = list(candidates)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(candidates, n)


def course_actual_answer(item: dict, course: dict | None) -> Any:
    if not course:
        return None
    qtype = item.get("type")
    if qtype == "midterm_points":
        return course.get("midterm_points")
    if qtype == "final_points":
        return course.get("final_points")
    if qtype == "project_points":
        return course.get("project_points")
    if qtype == "has_project":
        return bool(course.get("has_project"))
    if qtype in {"who_teaches", "is_teacher_yes", "is_teacher_no"}:
        return list(course.get("teachers", []))
    if qtype == "is_summer":
        return course.get("semester")
    if qtype == "credits":
        return course.get("credits")
    if qtype == "textbook":
        return course.get("textbook")
    if qtype == "department":
        return course.get("department")
    if qtype == "code_expansion":
        return course.get("course_name")
    if qtype == "code_lookup":
        return course.get("code")
    return item.get("expected_answer")


def group_actual_answer(item: dict, group: dict | None) -> Any:
    if not group:
        return None
    qtype = item.get("type")
    if qtype in {"works_on_topic_yes", "works_on_topic_no"}:
        return list(group.get("research_interests", []))
    if qtype == "cooperate_with_yes":
        return list(group.get("cooperation", []))
    if qtype == "contact_person":
        return group.get("contact_person")
    if qtype == "group_work_summary":
        return {
            "overview": list(group.get("overview", [])),
            "research_interests": list(group.get("research_interests", [])),
        }
    if qtype in {"member_of_group_yes", "member_of_group_no"}:
        return list(group.get("member_names", []))
    if qtype == "member_role_yes":
        return list(group.get("members", []))
    if qtype in {"working_on_project_yes", "working_on_project_no"}:
        return list(group.get("project_titles", []))
    return item.get("expected_answer")


def personnel_actual_answer(item: dict, person: dict | None, award_winners: dict[str, list[str]] | None = None) -> Any:
    qtype = item.get("type")

    if qtype == "award_who":
        award = item.get("award")
        if award and award_winners is not None:
            return list(award_winners.get(award, []))
        return item.get("expected_answer")

    if not person:
        return None

    if qtype in {"is_part_of_yes", "is_part_of_no"}:
        return [o.get("organization") for o in person.get("organizations", []) if o.get("organization")]
    if qtype in {"email", "example_milan_email"}:
        return person.get("email")
    if qtype in {"phone", "example_milan_phone"}:
        return person.get("phone")
    if qtype in {"office_yes", "office_no", "example_milan_office"}:
        return person.get("room")
    if qtype in {"project_running_yes", "project_running_no"}:
        return [p.get("title") for p in person.get("running_projects", []) if p.get("title")]
    if qtype in {"project_completed_yes", "project_completed_no"}:
        return [p.get("title") for p in person.get("completed_projects", []) if p.get("title")]
    if qtype in {"award_did_yes", "award_did_no"}:
        return list(person.get("awards", []))
    if qtype in {"consulting_hours", "example_santosh_consulting"}:
        return person.get("consulting_hours")
    if qtype in {"teaches", "example_santosh_teaches"}:
        return list(person.get("taught_courses", []))
    return item.get("expected_answer")


def pick_short_text(items: list[str]) -> str | None:
    if not items:
        return None
    # Prefer short, informative entries.
    ranked = sorted((clean_text(x) for x in items if clean_text(x)), key=lambda x: len(x))
    if not ranked:
        return None
    best = ranked[0]
    # keep answer short
    if len(best) > 140:
        best = best.split(".")[0].strip()
    return best[:160]


def concise_summary(text: str, max_len: int = 96) -> str | None:
    text = clean_text(text)
    if not text:
        return None

    # Prefer full first sentence. If it's too long, skip instead of truncating.
    sentence_parts = [p.strip() for p in re.split(r"[.;]", text) if p.strip()]
    if sentence_parts:
        first_sentence = sentence_parts[0]
        if len(first_sentence) <= max_len:
            return first_sentence

    # Fallback: keep full text only if already short.
    if len(text) <= max_len:
        return text

    # No complete concise snippet without truncation -> caller should skip.
    return None


def extract_partner_phrase(text: str) -> str:
    text = clean_text(text)
    if not text:
        return "the listed partner"

    patterns = [
        r"with\s+(?:the\s+)?([^,.\-]+)",
        r"cooperation\s+with\s+(?:the\s+)?([^,.\-]+)",
        r"collaboration\s+with\s+(?:the\s+)?([^,.\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            phrase = clean_text(m.group(1))
            if phrase:
                return phrase[:90]

    # fallback: first 7 words
    return " ".join(text.split()[:7])


def person_query_name(full_name: str) -> str:
    base = clean_text((full_name or "").split(",")[0])
    tokens = base.split()
    if len(tokens) == 2:
        # Source uses "Surname Given" in many files; questions are more natural as "given surname".
        return f"{tokens[1]} {tokens[0]}"
    return base


def parse_project_status(details: str) -> str | None:
    details_l = (details or "").lower()
    if "running" in details_l:
        return "running"
    if "completed" in details_l:
        return "completed"
    return None


def person_record(profile_path: Path) -> dict:
    data = load_json(profile_path)
    contact = data.get("contact", {})
    details = contact.get("details", {})
    roles = data.get("roles", [])
    teaching = data.get("teaching", {})
    projects = data.get("projects", [])
    curriculum = data.get("curriculum", {})

    person_name = person_query_name(data.get("name", profile_path.stem))

    organizations = []
    for r in roles:
        if not isinstance(r, dict):
            continue
        org = clean_text(r.get("organization", ""))
        role = clean_text(r.get("role", ""))
        if org:
            organizations.append({"organization": org, "role": role})

    running_projects = []
    completed_projects = []
    for p in projects:
        if not isinstance(p, dict):
            continue
        title = clean_text(p.get("title", ""))
        p_details = clean_text(p.get("details", ""))
        if not title:
            continue
        status = parse_project_status(p_details)
        item = {"title": title, "status": status}
        if status == "running":
            running_projects.append(item)
        elif status == "completed":
            completed_projects.append(item)

    awards = [clean_text(a) for a in curriculum.get("awards", []) if clean_text(a)]

    taught_courses = []
    for key in ("lectured_courses", "guaranteed_courses"):
        for c in teaching.get(key, []) or []:
            if not isinstance(c, dict):
                continue
            title = clean_text(c.get("title", ""))
            abbr = clean_text(c.get("abbreviation", ""))
            if title:
                taught_courses.append(f"{title} ({abbr})" if abbr else title)

    # de-duplicate while preserving order
    dedup_taught = []
    seen = set()
    for t in taught_courses:
        if t not in seen:
            seen.add(t)
            dedup_taught.append(t)

    return {
        "name": person_name,
        "email": clean_text(details.get("e_mail", "")) or None,
        "phone": clean_text(details.get("work_phone", "")) or None,
        "room": clean_text(details.get("room", "")) or None,
        "organizations": organizations,
        "consulting_hours": clean_text(teaching.get("consulting_hours", "")) or None,
        "taught_courses": dedup_taught,
        "running_projects": running_projects,
        "completed_projects": completed_projects,
        "awards": awards,
    }


def generate_personnel_candidate_questions(profiles: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []

    all_orgs = sorted({o["organization"] for p in profiles for o in p["organizations"]})
    all_names = sorted({p["name"] for p in profiles if p["name"]})
    all_rooms = sorted({p["room"] for p in profiles if p["room"]})
    all_awards = sorted({a for p in profiles for a in p["awards"]})
    all_running_projects = sorted({pr["title"] for p in profiles for pr in p["running_projects"]})
    all_completed_projects = sorted({pr["title"] for p in profiles for pr in p["completed_projects"]})

    award_to_people: dict[str, list[str]] = {}
    for p in profiles:
        for a in p["awards"]:
            award_to_people.setdefault(a, []).append(p["name"])

    q_templates = {
        "is_part_of": [
            "Is {person} part of {organization}?",
            "Does {person} belong to {organization}?",
            "Is {person} a member of {organization}?",
        ],
        "email": [
            "What is the email for {person}?",
            "What is {person}'s email address?",
            "How can I email {person}?",
        ],
        "phone": [
            "What is the phone number of {person}?",
            "What is {person}'s work phone number?",
            "How can I call {person}?",
        ],
        "office_yesno": [
            "Does {person} work at {room}?",
            "Is {person}'s office {room}?",
            "Is {person} located in {room}?",
        ],
        "project_running": [
            "Is {person} working on {project} project?",
            "Does {person} work on {project} project?",
            "Is {person} currently involved in {project} project?",
        ],
        "project_completed": [
            "Has {person} worked on {project} project?",
            "Did {person} work on {project} project?",
            "Was {person} involved in {project} project?",
        ],
        "award_who": [
            "Who won {award}?",
            "Who is listed as winner of {award}?",
            "Who received {award}?",
        ],
        "award_did": [
            "Did {person} win {award}?",
            "Has {person} received {award}?",
            "Was {award} awarded to {person}?",
        ],
        "consulting_hours": [
            "What are the consulting hours for {person}?",
            "When are consulting hours of {person}?",
            "What is the consultation schedule of {person}?",
        ],
        "teaches": [
            "What does {person} teach?",
            "Which courses does {person} teach?",
            "What courses are taught by {person}?",
        ],
    }

    for p in profiles:
        person = p["name"]

        # Role membership yes/no
        if p["organizations"]:
            org_yes = rng.choice(p["organizations"])["organization"]
            candidates.append({
                "question": pick_template(rng, q_templates["is_part_of"], person=person, organization=org_yes),
                "expected_answer": "Yes",
                "answer_field": "roles[].organization",
                "person": person,
                "type": "is_part_of_yes",
            })

            org_negatives = [o for o in all_orgs if o not in {x["organization"] for x in p["organizations"]}]
            if org_negatives:
                org_no = rng.choice(org_negatives)
                candidates.append({
                    "question": pick_template(rng, q_templates["is_part_of"], person=person, organization=org_no),
                    "expected_answer": "No",
                    "answer_field": "roles[].organization",
                    "person": person,
                    "type": "is_part_of_no",
                })

        # Contact details
        if p["email"]:
            candidates.append({
                "question": pick_template(rng, q_templates["email"], person=person),
                "expected_answer": p["email"],
                "answer_field": "contact.details.e_mail",
                "person": person,
                "type": "email",
            })

        if p["phone"]:
            candidates.append({
                "question": pick_template(rng, q_templates["phone"], person=person),
                "expected_answer": p["phone"],
                "answer_field": "contact.details.work_phone",
                "person": person,
                "type": "phone",
            })

        if p["room"]:
            candidates.append({
                "question": pick_template(rng, q_templates["office_yesno"], person=person, room=p["room"]),
                "expected_answer": "Yes",
                "answer_field": "contact.details.room",
                "person": person,
                "type": "office_yes",
            })

            room_negatives = [r for r in all_rooms if r != p["room"]]
            if room_negatives:
                room_no = rng.choice(room_negatives)
                candidates.append({
                    "question": pick_template(rng, q_templates["office_yesno"], person=person, room=room_no),
                    "expected_answer": "No",
                    "answer_field": "contact.details.room",
                    "person": person,
                    "type": "office_no",
                })

        # Projects (wording depends on running/completed status)
        if p["running_projects"]:
            pr_yes = rng.choice(p["running_projects"])["title"]
            candidates.append({
                "question": pick_template(rng, q_templates["project_running"], person=person, project=pr_yes),
                "expected_answer": "Yes",
                "answer_field": "projects[].title + projects[].details(status=running)",
                "person": person,
                "type": "project_running_yes",
            })

            neg_pool = [t for t in all_running_projects if t not in {x["title"] for x in p["running_projects"]}]
            if neg_pool:
                pr_no = rng.choice(neg_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["project_running"], person=person, project=pr_no),
                    "expected_answer": "No",
                    "answer_field": "projects[].title + projects[].details(status=running)",
                    "person": person,
                    "type": "project_running_no",
                })

        if p["completed_projects"]:
            pc_yes = rng.choice(p["completed_projects"])["title"]
            candidates.append({
                "question": pick_template(rng, q_templates["project_completed"], person=person, project=pc_yes),
                "expected_answer": "Yes",
                "answer_field": "projects[].title + projects[].details(status=completed)",
                "person": person,
                "type": "project_completed_yes",
            })

            neg_pool = [t for t in all_completed_projects if t not in {x["title"] for x in p["completed_projects"]}]
            if neg_pool:
                pc_no = rng.choice(neg_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["project_completed"], person=person, project=pc_no),
                    "expected_answer": "No",
                    "answer_field": "projects[].title + projects[].details(status=completed)",
                    "person": person,
                    "type": "project_completed_no",
                })

        # Awards
        if p["awards"]:
            award_yes = rng.choice(p["awards"])
            candidates.append({
                "question": pick_template(rng, q_templates["award_did"], person=person, award=award_yes),
                "expected_answer": "Yes",
                "answer_field": "curriculum.awards[]",
                "person": person,
                "award": award_yes,
                "type": "award_did_yes",
            })

            award_no_pool = [a for a in all_awards if a not in p["awards"]]
            if award_no_pool:
                award_no = rng.choice(award_no_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["award_did"], person=person, award=award_no),
                    "expected_answer": "No",
                    "answer_field": "curriculum.awards[]",
                    "person": person,
                    "award": award_no,
                    "type": "award_did_no",
                })

        if p["consulting_hours"]:
            candidates.append({
                "question": pick_template(rng, q_templates["consulting_hours"], person=person),
                "expected_answer": p["consulting_hours"],
                "answer_field": "teaching.consulting_hours",
                "person": person,
                "type": "consulting_hours",
            })

        if p["taught_courses"]:
            ans = ", ".join(p["taught_courses"][:4])
            candidates.append({
                "question": pick_template(rng, q_templates["teaches"], person=person),
                "expected_answer": ans,
                "answer_field": "teaching.lectured_courses[]/teaching.guaranteed_courses[]",
                "person": person,
                "type": "teaches",
            })

    for award, people in award_to_people.items():
        if not people:
            continue
        winner = rng.choice(sorted(set(people)))
        candidates.append({
            "question": pick_template(rng, q_templates["award_who"], award=award),
            "expected_answer": winner,
            "answer_field": "curriculum.awards[] across personnel_profiles",
            "person": winner,
            "award": award,
            "type": "award_who",
        })

    # Guaranteed concrete examples requested by user (included only when data exists).
    by_name = {p["name"].lower(): p for p in profiles}

    milan = by_name.get("milan ceska")
    if milan:
        if milan.get("email"):
            candidates.append({
                "question": "What is the email for milan ceska?",
                "expected_answer": milan["email"],
                "answer_field": "contact.details.e_mail",
                "person": milan["name"],
                "type": "example_milan_email",
            })
        if milan.get("room"):
            candidates.append({
                "question": f"Does milan ceska work at {milan['room']}?",
                "expected_answer": "Yes",
                "answer_field": "contact.details.room",
                "person": milan["name"],
                "type": "example_milan_office",
            })
        if milan.get("phone"):
            candidates.append({
                "question": "What is the phone number of milan ceska?",
                "expected_answer": milan["phone"],
                "answer_field": "contact.details.work_phone",
                "person": milan["name"],
                "type": "example_milan_phone",
            })

    santosh = by_name.get("santosh kesiraju")
    if santosh:
        if santosh.get("consulting_hours"):
            candidates.append({
                "question": "What are the consulting hours for santosh kesiraju?",
                "expected_answer": santosh["consulting_hours"],
                "answer_field": "teaching.consulting_hours",
                "person": santosh["name"],
                "type": "example_santosh_consulting",
            })
        if santosh.get("taught_courses"):
            candidates.append({
                "question": "What does santosh kesiraju teach?",
                "expected_answer": ", ".join(santosh["taught_courses"][:4]),
                "answer_field": "teaching.lectured_courses[]/teaching.guaranteed_courses[]",
                "person": santosh["name"],
                "type": "example_santosh_teaches",
            })

    unique = {}
    for item in candidates:
        unique[item["question"]] = item
    return list(unique.values())


def personnel_questions_structred(
    input_dir: str = "extracted_data_clean/fit/personnel_profiles",
    output_file: str = "FIT_RAG_Benchmark/personnel/structured.json",
    num_questions: int = 50,
    seed: int = 1337,
) -> dict:
    profile_paths = sorted(Path(input_dir).glob("*.json"))
    profiles = [person_record(path) for path in profile_paths]
    profiles_by_name = {p["name"]: p for p in profiles}
    award_winners: dict[str, list[str]] = {}
    for p in profiles:
        for award in p.get("awards", []):
            award_winners.setdefault(award, []).append(p["name"])

    candidates = generate_personnel_candidate_questions(profiles, seed=seed)

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    quotas = {
        "example_milan_email": 1,
        "example_milan_office": 1,
        "example_milan_phone": 1,
        "example_santosh_consulting": 1,
        "example_santosh_teaches": 1,
        "is_part_of_yes": 4,
        "is_part_of_no": 4,
        "email": 4,
        "phone": 4,
        "office_yes": 3,
        "office_no": 2,
        "project_running_yes": 3,
        "project_running_no": 3,
        "project_completed_yes": 3,
        "project_completed_no": 3,
        "award_who": 2,
        "award_did_yes": 3,
        "award_did_no": 2,
        "consulting_hours": 3,
        "teaches": 2,
    }

    sampled: list[dict] = []
    used_questions = set()

    for t, q in quotas.items():
        picked = 0
        for item in by_type.get(t, []):
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            picked += 1
            if picked >= q:
                break

    if len(sampled) < num_questions:
        pool = list(candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]
    sampled = sorted(sampled, key=lambda x: (x["person"], x["type"], x["question"]))

    items = []
    for i, item in enumerate(sampled, start=1):
        item_out = dict(item)
        item_out["actual_answer"] = personnel_actual_answer(
            item,
            profiles_by_name.get(item.get("person")),
            award_winners=award_winners,
        )
        items.append({"id": f"personnel_struct_{i:03d}", **item_out})

    output = {
        "dataset": "FIT_personnel_structured_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "description": "Structured short-answer personnel benchmark with role, contact, project, award, and teaching questions.",
        "items": items,
    }
    save_json(output, Path(output_file))
    return output


def project_record(project_path: Path) -> dict:
    data = load_json(project_path)
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}

    title = clean_text(data.get("title", "")) or clean_text(data.get("project", "")) or project_path.stem
    agency = clean_text(metadata.get("agency", "")) or None
    program = clean_text(metadata.get("program", "")) or None
    period = clean_text(metadata.get("project_period", "")) or None
    code = clean_text(metadata.get("code", "")) or None

    return {
        "title": title,
        "agency": agency,
        "program": program,
        "period": period,
        "code": code,
    }


def generate_project_candidate_questions(projects: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates: list[dict] = []

    agencies = sorted({p["agency"] for p in projects if p.get("agency")})

    q_templates = {
        "funded_by": [
            "Is this {project} funded by {agency}?",
            "Is project {project} funded by {agency}?",
            "Is {project} financed by {agency}?",
        ],
        "agency_funding": [
            "Is {agency} funding this project {project}?",
            "Is {agency} the funder of {project}?",
            "Does funding for {project} come from {agency}?",
        ],
        "who_funding": [
            "Who is funding this project {project}?",
            "Which agency is funding {project}?",
            "What agency funds project {project}?",
        ],
        "program": [
            "What program is this project {project} part of?",
            "Which program does {project} belong to?",
            "Under what program is {project} listed?",
        ],
        "period": [
            "What is the period for this project {project}?",
            "What is the project period of {project}?",
            "What is the duration of project {project}?",
        ],
        "code": [
            "What is the project code for {project}?",
            "Which code is assigned to project {project}?",
            "What is the code of project {project}?",
        ],
    }

    for p in projects:
        title = p["title"]

        if p["agency"]:
            candidates.append({
                "question": pick_template(rng, q_templates["funded_by"], project=title, agency=p["agency"]),
                "expected_answer": "Yes",
                "answer_field": "metadata.agency",
                "project": title,
                "type": "funded_by_yes",
                "actual_answer": p["agency"],
            })
            candidates.append({
                "question": pick_template(rng, q_templates["agency_funding"], project=title, agency=p["agency"]),
                "expected_answer": "Yes",
                "answer_field": "metadata.agency",
                "project": title,
                "type": "agency_funding_yes",
                "actual_answer": p["agency"],
            })
            candidates.append({
                "question": pick_template(rng, q_templates["who_funding"], project=title),
                "expected_answer": p["agency"],
                "answer_field": "metadata.agency",
                "project": title,
                "type": "who_funding",
                "actual_answer": p["agency"],
            })

            wrong_agencies = [a for a in agencies if a != p["agency"]]
            if wrong_agencies:
                wrong = rng.choice(wrong_agencies)
                candidates.append({
                    "question": pick_template(rng, q_templates["funded_by"], project=title, agency=wrong),
                    "expected_answer": "No",
                    "answer_field": "metadata.agency",
                    "project": title,
                    "type": "funded_by_no",
                    "actual_answer": p["agency"],
                })
                candidates.append({
                    "question": pick_template(rng, q_templates["agency_funding"], project=title, agency=wrong),
                    "expected_answer": "No",
                    "answer_field": "metadata.agency",
                    "project": title,
                    "type": "agency_funding_no",
                    "actual_answer": p["agency"],
                })

        if p["program"]:
            candidates.append({
                "question": pick_template(rng, q_templates["program"], project=title),
                "expected_answer": p["program"],
                "answer_field": "metadata.program",
                "project": title,
                "type": "project_program",
                "actual_answer": p["program"],
            })

        if p["period"]:
            candidates.append({
                "question": pick_template(rng, q_templates["period"], project=title),
                "expected_answer": p["period"],
                "answer_field": "metadata.project_period",
                "project": title,
                "type": "project_period",
                "actual_answer": p["period"],
            })

        if p["code"]:
            candidates.append({
                "question": pick_template(rng, q_templates["code"], project=title),
                "expected_answer": p["code"],
                "answer_field": "metadata.code",
                "project": title,
                "type": "project_code",
                "actual_answer": p["code"],
            })

    unique = {}
    for item in candidates:
        unique[item["question"]] = item
    return list(unique.values())


def _sample_with_project_diversity(pool: list[dict], quota: int, used_questions: set[str]) -> list[dict]:
    picked: list[dict] = []
    used_projects: set[str] = set()

    for item in pool:
        if item["question"] in used_questions:
            continue
        project = item.get("project")
        if project in used_projects:
            continue
        picked.append(item)
        used_questions.add(item["question"])
        if project:
            used_projects.add(project)
        if len(picked) >= quota:
            return picked

    for item in pool:
        if item["question"] in used_questions:
            continue
        picked.append(item)
        used_questions.add(item["question"])
        if len(picked) >= quota:
            break

    return picked


def project_questions_structred(
    input_dir: str = "extracted_data_clean/fit/projects",
    output_file: str = "FIT_RAG_Benchmark/projects/structured.json",
    num_questions: int = 50,
    seed: int = 1337,
) -> dict:
    project_paths = sorted(Path(input_dir).glob("*.json"))
    projects = [project_record(path) for path in project_paths]

    candidates = generate_project_candidate_questions(projects, seed=seed)
    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    quotas = {
        "funded_by_yes": 8,
        "funded_by_no": 8,
        "agency_funding_yes": 8,
        "agency_funding_no": 8,
        "who_funding": 7,
        "project_program": 5,
        "project_period": 4,
        "project_code": 2,
    }

    sampled: list[dict] = []
    used_questions: set[str] = set()

    for t, q in quotas.items():
        pool = by_type.get(t, [])
        sampled.extend(_sample_with_project_diversity(pool, q, used_questions))

    if len(sampled) < num_questions:
        pool = list(candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]
    sampled = sorted(sampled, key=lambda x: (x["project"], x["type"], x["question"]))

    items = []
    for i, item in enumerate(sampled, start=1):
        items.append({"id": f"projects_struct_{i:03d}", **item})

    output = {
        "dataset": "FIT_projects_structured_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "description": "Structured short-answer project benchmark with funding, program, period, and code questions.",
        "items": items,
    }
    save_json(output, Path(output_file))
    return output


def extract_bibtex_authors(bibtex_text: str) -> list[str]:
    text = bibtex_text or ""
    m = re.search(r'author\s*=\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    parts = [clean_text(p) for p in raw.split(" and ") if clean_text(p)]

    names: list[str] = []
    for p in parts:
        p = p.replace("{", "").replace("}", "")
        p = clean_text(p)
        if not p:
            continue
        # Handle forms like "SURNAME, I." or "Jan Peciva"
        if "," in p:
            chunks = [clean_text(x) for x in p.split(",") if clean_text(x)]
            if len(chunks) >= 2:
                surname = chunks[0]
                given = chunks[1].split()[0] if chunks[1].split() else ""
                candidate = clean_text(f"{given} {surname}")
                if candidate and len(candidate) >= 3:
                    names.append(candidate)
                    continue
        if len(p) >= 3:
            names.append(p)

    deduped: list[str] = []
    seen = set()
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


def extract_authors_from_sections(authors_text: str) -> list[str]:
    text = clean_text(authors_text or "")
    if not text:
        return []

    # Remove department tags and titles to keep cleaner names.
    text = re.sub(r"\([^\)]*\)", "", text)
    text = re.sub(r"\b(Ing\.|Ph\.D\.|doc\.|prof\.|RNDr\.|Mgr\.|Dr\.|CSc\.)\b", "", text, flags=re.IGNORECASE)
    text = text.replace(";", " ")

    # Capture simple two-token names.
    candidates = re.findall(r"\b[A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+\b", text)
    deduped: list[str] = []
    seen = set()
    for c in candidates:
        c = clean_text(c)
        key = c.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def extract_publication_projects(projects_text: str) -> list[str]:
    text = clean_text(projects_text or "")
    if not text:
        return []

    # Usually: "Project Name , AGENCY, PROGRAM, ..."
    segments = [clean_text(s) for s in re.split(r"\s*;\s*|\s*\|\s*", text) if clean_text(s)]
    names = []
    for seg in segments:
        name = clean_text(seg.split(",")[0])
        if name:
            names.append(name)

    deduped: list[str] = []
    seen = set()
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    return deduped


def publication_record(pub_path: Path) -> dict:
    data = load_json(pub_path)
    sections = data.get("sections", {}) if isinstance(data.get("sections"), dict) else {}

    title = clean_text(data.get("title", "")) or pub_path.stem
    year = clean_text(data.get("year", "")) or clean_text(sections.get("published", {}).get("text", ""))
    ptype = clean_text(sections.get("type", {}).get("text", ""))
    publisher = clean_text(sections.get("publisher", {}).get("text", ""))

    bibtex = clean_text(sections.get("bibtex", {}).get("text", ""))
    authors_section = clean_text(sections.get("authors", {}).get("text", ""))
    authors = extract_bibtex_authors(bibtex) or extract_authors_from_sections(authors_section)
    authors = [a for a in authors if clean_text(a) and len(clean_text(a)) >= 3]

    projects = extract_publication_projects(sections.get("projects", {}).get("text", ""))

    return {
        "title": title,
        "year": year,
        "type": ptype,
        "is_conference": "conference" in ptype.lower(),
        "publisher": publisher or None,
        "authors": authors,
        "projects": projects,
    }


def generate_publication_candidate_questions(publications: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates: list[dict] = []

    all_authors = sorted({a for p in publications for a in p.get("authors", [])})
    all_projects = sorted({pr for p in publications for pr in p.get("projects", [])})

    q_templates = {
        "conference": [
            "Is this paper {title} a conference paper?",
            "Is {title} a conference paper?",
            "Is publication {title} of type conference paper?",
        ],
        "author": [
            "Is {person} an author in this work {title}?",
            "Is {person} listed as an author of {title}?",
            "Did {person} co-author the paper {title}?",
        ],
        "year_2025": [
            "Was this paper {title} published in 2025?",
            "Is {title} a 2025 publication?",
            "Was {title} published in the year 2025?",
        ],
        "project": [
            "Was this paper {title} published as part of this project {project}?",
            "Is {title} linked to project {project}?",
            "Was {title} produced within project {project}?",
        ],
        "publisher": [
            "Who is the publisher for this paper {title}?",
            "What is the publisher of {title}?",
            "Which publisher released {title}?",
        ],
    }

    for p in publications:
        title = p["title"]

        candidates.append({
            "question": pick_template(rng, q_templates["conference"], title=title),
            "expected_answer": "Yes" if p["is_conference"] else "No",
            "answer_field": "sections.type.text",
            "publication_title": title,
            "type": "is_conference_paper",
            "actual_answer": p.get("type"),
        })

        if p["authors"]:
            author_pool = [a for a in p["authors"] if clean_text(a)]
            if not author_pool:
                author_pool = []
            if author_pool:
                author_yes = rng.choice(author_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["author"], person=author_yes, title=title),
                    "expected_answer": "Yes",
                    "answer_field": "sections.authors.text / sections.bibtex.text",
                    "publication_title": title,
                    "type": "is_author_yes",
                    "person": author_yes,
                    "actual_answer": list(author_pool),
                })

                negatives = [a for a in all_authors if a not in author_pool]
                if negatives:
                    author_no = rng.choice(negatives)
                    candidates.append({
                        "question": pick_template(rng, q_templates["author"], person=author_no, title=title),
                        "expected_answer": "No",
                        "answer_field": "sections.authors.text / sections.bibtex.text",
                        "publication_title": title,
                        "type": "is_author_no",
                        "person": author_no,
                        "actual_answer": list(author_pool),
                    })

        candidates.append({
            "question": pick_template(rng, q_templates["year_2025"], title=title),
            "expected_answer": "Yes" if p.get("year") == "2025" else "No",
            "answer_field": "year / sections.published.text",
            "publication_title": title,
            "type": "published_in_2025",
            "actual_answer": p.get("year"),
        })

        if p["projects"]:
            project_yes = rng.choice(p["projects"])
            candidates.append({
                "question": pick_template(rng, q_templates["project"], title=title, project=project_yes),
                "expected_answer": "Yes",
                "answer_field": "sections.projects.text",
                "publication_title": title,
                "type": "published_part_of_project_yes",
                "project": project_yes,
                "actual_answer": list(p["projects"]),
            })

            negatives = [pr for pr in all_projects if pr not in p["projects"]]
            if negatives:
                project_no = rng.choice(negatives)
                candidates.append({
                    "question": pick_template(rng, q_templates["project"], title=title, project=project_no),
                    "expected_answer": "No",
                    "answer_field": "sections.projects.text",
                    "publication_title": title,
                    "type": "published_part_of_project_no",
                    "project": project_no,
                    "actual_answer": list(p["projects"]),
                })

        if p["publisher"]:
            candidates.append({
                "question": pick_template(rng, q_templates["publisher"], title=title),
                "expected_answer": p["publisher"],
                "answer_field": "sections.publisher.text",
                "publication_title": title,
                "type": "publisher",
                "actual_answer": p["publisher"],
            })

    unique = {}
    for item in candidates:
        unique[item["question"]] = item
    return list(unique.values())


def publication_questions_structred(
    input_dir: str = "extracted_data_clean/fit/publications",
    output_file: str = "FIT_RAG_Benchmark/publications/structured.json",
    num_questions: int = 50,
    seed: int = 1337,
) -> dict:
    pub_paths = sorted(Path(input_dir).glob("*.json"))
    publications = [publication_record(path) for path in pub_paths]

    candidates = generate_publication_candidate_questions(publications, seed=seed)
    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    quotas = {
        "is_conference_paper": 14,
        "is_author_yes": 10,
        "is_author_no": 8,
        "published_in_2025": 10,
        "published_part_of_project_yes": 3,
        "published_part_of_project_no": 2,
        "publisher": 3,
    }

    sampled: list[dict] = []
    used_questions = set()

    for t, q in quotas.items():
        pool = by_type.get(t, [])
        picked = 0
        used_pubs: set[str] = set()

        for item in pool:
            if item["question"] in used_questions:
                continue
            pub_title = item.get("publication_title")
            if pub_title in used_pubs:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if pub_title:
                used_pubs.add(pub_title)
            picked += 1
            if picked >= q:
                break

        if picked < q:
            for item in pool:
                if item["question"] in used_questions:
                    continue
                sampled.append(item)
                used_questions.add(item["question"])
                picked += 1
                if picked >= q:
                    break

    if len(sampled) < num_questions:
        pool = list(candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]
    sampled = sorted(sampled, key=lambda x: (x["publication_title"], x["type"], x["question"]))

    items = []
    for i, item in enumerate(sampled, start=1):
        items.append({"id": f"publications_struct_{i:03d}", **item})

    output = {
        "dataset": "FIT_publications_structured_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "description": "Structured short-answer publication benchmark with type, author, year, project, and publisher questions.",
        "items": items,
    }
    save_json(output, Path(output_file))
    return output


def group_record(group_path: Path) -> dict:
    data = load_json(group_path)
    about = data.get("about", {})
    team = data.get("team", {})
    projects = data.get("projects", [])

    members = []
    member_names = []
    for role_key, entries in team.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            name = clean_text(e.get("name", ""))
            position = clean_text(e.get("position", ""))
            if not name:
                continue
            members.append({"name": name, "role_key": role_key, "position": position})
            member_names.append(name)

    research_interests = [clean_text(x) for x in about.get("research_interests", []) if clean_text(x)]
    cooperation = [clean_text(x) for x in about.get("cooperation", []) if clean_text(x)]
    overview = [clean_text(x) for x in about.get("overview", []) if clean_text(x)]
    project_titles = [
        clean_text(p.get("title", ""))
        for p in projects
        if isinstance(p, dict) and clean_text(p.get("title", ""))
    ]

    principal = team.get("principal_researcher", [])
    contact_person = None
    if isinstance(principal, list) and principal:
        first = principal[0]
        if isinstance(first, dict):
            contact_person = clean_text(first.get("name", "")) or None

    if not contact_person and member_names:
        contact_person = member_names[0]

    return {
        "group": clean_text(data.get("group", group_path.stem)),
        "research_interests": research_interests,
        "cooperation": cooperation,
        "overview": overview,
        "contact_person": contact_person,
        "members": members,
        "member_names": member_names,
        "project_titles": project_titles,
    }


def generate_group_candidate_questions(groups: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = []

    all_member_names = sorted({n for g in groups for n in g["member_names"]})
    all_project_titles = sorted({t for g in groups for t in g["project_titles"]})
    all_interest_topics = sorted({t for g in groups for t in g["research_interests"]})

    q_templates = {
        "works_on_topic": [
            "Does {group} work on {topic}?",
            "Is {topic} a research interest of {group}?",
            "Does the {group} group focus on {topic}?",
        ],
        "cooperate_with": [
            "Does {group} cooperate with {partner}?",
            "Is {partner} listed as a cooperation partner of {group}?",
            "Does the {group} group collaborate with {partner}?",
        ],
        "contact_person": [
            "Who is the contact person for {group}?",
            "Who should I contact in {group}?",
            "Who is the main contact in {group}?",
        ],
        "group_work_summary": [
            "What does {group} work on?",
            "What is {group} mainly focused on?",
            "What is the primary focus of {group}?",
        ],
        "member_of_group": [
            "Is {person} part of {group}?",
            "Does {person} belong to {group}?",
            "Is {person} in the {group} group?",
        ],
        "member_role": [
            "Is {person} a {role} in {group}?",
            "Does {person} have role {role} in {group}?",
            "Is {person}'s role {role} in {group}?",
        ],
        "working_on_project": [
            "Is {group} working on {project}?",
            "Does {group} have a project titled {project}?",
            "Is {project} listed under projects of {group}?",
        ],
    }

    for g in groups:
        group_name = g["group"]

        # 1) Does group work on topic? (yes)
        if g["research_interests"]:
            topic_yes = rng.choice(g["research_interests"])
            candidates.append({
                "question": pick_template(rng, q_templates["works_on_topic"], group=group_name, topic=topic_yes),
                "expected_answer": "Yes",
                "answer_field": "about.research_interests",
                "group": group_name,
                "type": "works_on_topic_yes",
            })

            negatives = [t for t in all_interest_topics if t not in g["research_interests"]]
            if negatives:
                topic_no = rng.choice(negatives)
                candidates.append({
                    "question": pick_template(rng, q_templates["works_on_topic"], group=group_name, topic=topic_no),
                    "expected_answer": "No",
                    "answer_field": "about.research_interests",
                    "group": group_name,
                    "type": "works_on_topic_no",
                })

        # 2) cooperation partner questions
        if g["cooperation"]:
            coop_item = rng.choice(g["cooperation"])
            partner = extract_partner_phrase(coop_item)
            candidates.append({
                "question": pick_template(rng, q_templates["cooperate_with"], group=group_name, partner=partner),
                "expected_answer": "Yes",
                "answer_field": "about.cooperation",
                "group": group_name,
                "type": "cooperate_with_yes",
            })

        # 3) contact person
        if g["contact_person"]:
            candidates.append({
                "question": pick_template(rng, q_templates["contact_person"], group=group_name),
                "expected_answer": g["contact_person"],
                "answer_field": "team.principal_researcher[0].name",
                "group": group_name,
                "type": "contact_person",
            })

        # 4) what does this group work on? (short answer from overview)
        summary = None
        for txt in g["overview"]:
            summary = concise_summary(txt)
            if summary:
                break
        if not summary:
            for txt in g["research_interests"]:
                summary = concise_summary(txt)
                if summary:
                    break

        if summary:
            candidates.append({
                "question": pick_template(rng, q_templates["group_work_summary"], group=group_name),
                "expected_answer": summary,
                "answer_field": "about.overview/about.research_interests",
                "group": group_name,
                "type": "group_work_summary",
            })

        # 5) is person part of group?
        if g["member_names"]:
            yes_person = rng.choice(g["member_names"])
            candidates.append({
                "question": pick_template(rng, q_templates["member_of_group"], person=yes_person, group=group_name),
                "expected_answer": "Yes",
                "answer_field": "team.*.name",
                "group": group_name,
                "type": "member_of_group_yes",
            })

            no_pool = [n for n in all_member_names if n not in g["member_names"]]
            if no_pool:
                no_person = rng.choice(no_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["member_of_group"], person=no_person, group=group_name),
                    "expected_answer": "No",
                    "answer_field": "team.*.name",
                    "group": group_name,
                    "type": "member_of_group_no",
                })

        # 6) is person a role in group?
        if g["members"]:
            m = rng.choice(g["members"])
            role = m["position"] or m["role_key"].replace("_", " ")
            candidates.append({
                "question": pick_template(rng, q_templates["member_role"], person=m["name"], role=role, group=group_name),
                "expected_answer": "Yes",
                "answer_field": "team.*.position",
                "group": group_name,
                "type": "member_role_yes",
            })

        # 7) is this group working on project title?
        if g["project_titles"]:
            proj_yes = rng.choice(g["project_titles"])
            candidates.append({
                "question": pick_template(rng, q_templates["working_on_project"], group=group_name, project=proj_yes),
                "expected_answer": "Yes",
                "answer_field": "projects[].title",
                "group": group_name,
                "type": "working_on_project_yes",
            })

            proj_no_pool = [p for p in all_project_titles if p not in g["project_titles"]]
            if proj_no_pool:
                proj_no = rng.choice(proj_no_pool)
                candidates.append({
                    "question": pick_template(rng, q_templates["working_on_project"], group=group_name, project=proj_no),
                    "expected_answer": "No",
                    "answer_field": "projects[].title",
                    "group": group_name,
                    "type": "working_on_project_no",
                })

    unique = {}
    for item in candidates:
        unique[item["question"]] = item
    return list(unique.values())


def group_questions_structred(
    input_dir: str = "extracted_data_clean/fit/groups",
    output_file: str = "FIT_RAG_Benchmark/groups/structured.json",
    num_questions: int = 50,
    seed: int = 1337,
) -> dict:
    group_paths = sorted(Path(input_dir).glob("*.json"))
    groups = [group_record(path) for path in group_paths]
    groups_by_name = {g["group"]: g for g in groups}

    candidates = generate_group_candidate_questions(groups, seed=seed)

    # Balanced sampling by requested families for better benchmark coverage.
    # Family map: type -> quota
    quotas = {
        "works_on_topic_yes": 3,
        "works_on_topic_no": 3,
        "cooperate_with_yes": 6,
        "contact_person": 6,
        "group_work_summary": 6,
        "member_of_group_yes": 5,
        "member_of_group_no": 5,
        "member_role_yes": 8,
        "working_on_project_yes": 4,
        "working_on_project_no": 4,
    }

    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    sampled: list[dict] = []
    used_questions = set()

    for t, q in quotas.items():
        pool = by_type.get(t, [])
        picked = 0
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            picked += 1
            if picked >= q:
                break

    # Fill any remaining slots from all candidates deterministically.
    if len(sampled) < num_questions:
        pool = list(candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]
    sampled = sorted(sampled, key=lambda x: (x["group"], x["type"], x["question"]))

    items = []
    for i, item in enumerate(sampled, start=1):
        item_out = dict(item)
        item_out["actual_answer"] = group_actual_answer(item, groups_by_name.get(item.get("group")))
        items.append({"id": f"groups_struct_{i:03d}", **item_out})

    output = {
        "dataset": "FIT_groups_structured_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "description": "Structured short-answer group benchmark with paraphrased question templates and procedural answers.",
        "items": items,
    }
    save_json(output, Path(output_file))
    return output


def course_questions_structred(
    input_dir: str = "extracted_data_clean/fit/courses",
    output_file: str = "FIT_RAG_Benchmark/courses/structured.json",
    num_questions: int = 50,
    seed: int = 1337,
) -> dict:
    """
    Generate structured course benchmark questions.
    NOTE: Function name intentionally follows user-specified spelling.
    """
    course_paths = sorted(Path(input_dir).glob("*.json"))
    courses = [course_record(path) for path in course_paths]
    courses_by_code = {c["code"]: c for c in courses}

    candidates = generate_candidate_questions(courses, seed=seed)
    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = {}
    for c in candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    # Ensure at least a few code<->name questions are always present.
    quotas = {
        "code_expansion": 4,
        "code_lookup": 4,
    }

    sampled: list[dict] = []
    used_questions = set()

    for t, q in quotas.items():
        picked = 0
        for item in by_type.get(t, []):
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            picked += 1
            if picked >= q:
                break

    if len(sampled) < num_questions:
        pool = list(candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_questions:
                continue
            sampled.append(item)
            used_questions.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]

    # Stable ordering for deterministic file diffs.
    sampled = sorted(sampled, key=lambda x: (x["course_code"], x["type"], x["question"]))

    items = []
    for i, item in enumerate(sampled, start=1):
        item_out = dict(item)
        item_out["actual_answer"] = course_actual_answer(item, courses_by_code.get(item.get("course_code")))
        items.append(
            {
                "id": f"courses_struct_{i:03d}",
                **item_out,
            }
        )

    output = {
        "dataset": "FIT_courses_structured_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "description": "Structured short-answer course benchmark with paraphrased question templates and procedural answers.",
        "items": items,
    }

    save_json(output, Path(output_file))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FIT RAG benchmark test sets.")
    parser.add_argument("--target", choices=["courses", "groups", "personnel", "projects", "publications"], default="courses")
    parser.add_argument("--input_dir", type=str, default="extracted_data_clean/fit/courses")
    parser.add_argument("--output_file", type=str, default="FIT_RAG_Benchmark/courses/structured.json")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.target == "courses":
        result = course_questions_structred(
            input_dir=args.input_dir,
            output_file=args.output_file,
            num_questions=args.num_questions,
            seed=args.seed,
        )
    else:
        if args.target == "groups":
            input_dir = args.input_dir if args.input_dir != "extracted_data_clean/fit/courses" else "extracted_data_clean/fit/groups"
            output_file = args.output_file if args.output_file != "FIT_RAG_Benchmark/courses/structured.json" else "FIT_RAG_Benchmark/groups/structured.json"
            result = group_questions_structred(
                input_dir=input_dir,
                output_file=output_file,
                num_questions=args.num_questions,
                seed=args.seed,
            )
        elif args.target == "personnel":
            input_dir = args.input_dir if args.input_dir != "extracted_data_clean/fit/courses" else "extracted_data_clean/fit/personnel_profiles"
            output_file = args.output_file if args.output_file != "FIT_RAG_Benchmark/courses/structured.json" else "FIT_RAG_Benchmark/personnel/structured.json"
            result = personnel_questions_structred(
                input_dir=input_dir,
                output_file=output_file,
                num_questions=args.num_questions,
                seed=args.seed,
            )
        elif args.target == "projects":
            input_dir = args.input_dir if args.input_dir != "extracted_data_clean/fit/courses" else "extracted_data_clean/fit/projects"
            output_file = args.output_file if args.output_file != "FIT_RAG_Benchmark/courses/structured.json" else "FIT_RAG_Benchmark/projects/structured.json"
            result = project_questions_structred(
                input_dir=input_dir,
                output_file=output_file,
                num_questions=args.num_questions,
                seed=args.seed,
            )
        else:
            input_dir = args.input_dir if args.input_dir != "extracted_data_clean/fit/courses" else "extracted_data_clean/fit/publications"
            output_file = args.output_file if args.output_file != "FIT_RAG_Benchmark/courses/structured.json" else "FIT_RAG_Benchmark/publications/structured.json"
            result = publication_questions_structred(
                input_dir=input_dir,
                output_file=output_file,
                num_questions=args.num_questions,
                seed=args.seed,
            )

    print(f"Wrote {result['num_questions']} questions to {result['output_file']}")


if __name__ == "__main__":
    main()
