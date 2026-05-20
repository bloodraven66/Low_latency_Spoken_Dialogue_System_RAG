import argparse
import importlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512
    tensor_parallel_size: int = 8
    dtype = "bfloat16"  # or "fp16", "int8", etc., depending on vLLM support and model requirements


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def has_value(text: str | None) -> bool:
    return bool(clean_text(text))


def split_bullet_lines(text: str | None) -> list[str]:
    src = text or ""
    out: list[str] = []
    for raw in src.split("\n"):
        item = clean_text(re.sub(r"^[-*•]\s*", "", raw))
        if item:
            out.append(item)
    deduped: list[str] = []
    seen = set()
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return deduped


def topic_heading(topic: str) -> str:
    """Extract a concise heading from a syllabus topic line.

    Examples:
    - "10. Neurofeedback (Auditory Skills): This topic ..." -> "Neurofeedback (Auditory Skills)"
    - "Temporal database systems, introduction" -> unchanged
    """
    text = clean_text(topic)
    if not text:
        return text

    # Drop leading numeric list markers (e.g., "10.", "3)").
    text = re.sub(r"^\s*\d+\s*[\.)]\s*", "", text)

    # Prefer the part before a descriptive colon.
    if ":" in text:
        head = clean_text(text.split(":", 1)[0])
        if head:
            text = head

    # Prefer first sentence as heading when the line includes multiple sentences.
    if "." in text:
        first_sentence = clean_text(text.split(".", 1)[0])
        if first_sentence and len(first_sentence) >= 8:
            text = first_sentence

    # Final compact fallback for unusually long headings.
    if len(text) > 90 and "," in text:
        first_clause = clean_text(text.split(",", 1)[0])
        if first_clause and len(first_clause) >= 8:
            text = first_clause

    return text.rstrip(" ;,.-")


def course_record(course_path: Path) -> dict:
    data = load_json(course_path)
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    content = data.get("content", {}) if isinstance(data.get("content"), dict) else {}

    return {
        "code": clean_text(metadata.get("code")) or course_path.stem,
        "course_name": clean_text(metadata.get("course_name")) or None,
        "learning_objectives": clean_text(content.get("learning_objectives")) or None,
        "syllabus_of_lectures": clean_text(content.get("syllabus_of_lectures")) or None,
        "syllabus_topics": split_bullet_lines(content.get("syllabus_of_lectures")),
        "progress_assessment": clean_text(content.get("progress_assessment")) or None,
        "study_literature": clean_text(content.get("study_literature")) or None,
        "fundamental_literature": clean_text(content.get("fundamental_literature")) or None,
        "prerequisite_knowledge_and_skills": clean_text(content.get("prerequisite_knowledge_and_skills")) or None,
        "syllabus_of_seminars": clean_text(content.get("syllabus_of_seminars")) or None,
        "language_of_instruction": clean_text(content.get("language_of_instruction")) or None,
        "completion": clean_text(content.get("completion")) or None,
        "time_span": clean_text(content.get("time_span")) or None,
    }


class VLLMInferenceEngine:
    def __init__(self, config: LLMConfig):
        try:
            vllm = importlib.import_module("vllm")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "vLLM is required for --run_inference mode. Install it in the active environment first."
            ) from exc

        LLM = getattr(vllm, "LLM")
        SamplingParams = getattr(vllm, "SamplingParams")

        self.config = config
        self._sampling = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        self._llm = LLM(model=config.model_name, tensor_parallel_size=config.tensor_parallel_size, dtype=config.dtype, max_model_len=2048, enforce_eager=True, gpu_memory_utilization=0.80)

    def short_answer(self, question: str, field_name: str, field_text: str) -> str:
        system_prompt = (
            "You are a precise academic QA assistant.\n\n"
            "Task:\n"
            "Given a QUESTION and a FULL_TEXT (course content), generate the best possible answer using ONLY FULL_TEXT.\n\n"
            "Rules:\n"
            # "- Do not use outside knowledge.\n"
            "- Answer directly and concisely based on the content of the FULL_TEXT.\n"
            "- If the question is yes/no, return exactly 'Yes' or 'No' as the answer.\n"
            "- If the answer can be inferred from FULL_TEXT but is not explicitly stated, provide the best concise answer based on the information available.\n"
            # "- Do not invent facts.\n"
            "- Keep the answer short (max 2 sentences).\n\n"
            "Output format (strict JSON only):\n"
            "{\n"
            "  \"answer\": \"<final answer>\",\n"
            "  \"evidence\": \"<short quote/paraphrase from FULL_TEXT that supports answer>\"\n"
            "}"
        )

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"FIELD:\n{field_name}\n\n"
            f"FULL_TEXT:\n{field_text}\n\n"
            "Generate the output in the required JSON format."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tokenizer = self._llm.get_tokenizer()
        # Apply OLMo's native chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print(user_prompt)

        outputs = self._llm.generate([prompt], sampling_params=self._sampling)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

        raw = clean_text(text)
        print(raw)
        if not raw:
            return "Not available in provided text"

        # Prefer strict JSON answer if model follows the contract.
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                answer = clean_text(payload.get("answer"))
                if answer:
                    return answer
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object if model wrapped output in prose.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group(0))
                if isinstance(payload, dict):
                    answer = clean_text(payload.get("answer"))
                    if answer:
                        return answer
            except json.JSONDecodeError:
                pass

        # Final fallback: keep raw text concise and deterministic.
        return raw


def candidate_questions_for_course(course: dict, rng: random.Random) -> list[dict]:
    code = course["code"]
    course_ref = f"course {code}"
    questions: list[dict] = []

    if has_value(course["learning_objectives"]):
        questions.append(
            {
                "type": "learning_objectives",
                "question": f"What are the learning objectives of {course_ref}?",
                "answer_field": "content.learning_objectives",
                "field_payload": course["learning_objectives"],
                "course_code": code,
            }
        )

    if len(course["syllabus_topics"]) >= 3:
        topics = rng.sample(course["syllabus_topics"], 3)
        topic_headings = [topic_heading(t) for t in topics]
        topic_text = "; ".join(topic_headings)
        questions.append(
            {
                "type": "topics_yes",
                "question": f"Does {course_ref} cover these topics: {topic_text}?",
                "answer_field": "content.syllabus_of_lectures",
                "field_payload": course["syllabus_of_lectures"],
                "expected_answer": "Yes",
                "actual_answer": topics,
                "course_code": code,
            }
        )

    if has_value(course["progress_assessment"]):
        questions.append(
            {
                "type": "progress_assessment",
                "question": f"How is progress assessed in {course_ref}?",
                "answer_field": "content.progress_assessment",
                "field_payload": course["progress_assessment"],
                "course_code": code,
            }
        )

    literature = course["fundamental_literature"] or course["study_literature"]
    if has_value(literature):
        questions.append(
            {
                "type": "study_literature",
                "question": f"What literature is recommended for {course_ref}?",
                "answer_field": "content.fundamental_literature/content.study_literature",
                "field_payload": literature,
                "course_code": code,
            }
        )

    if has_value(course["prerequisite_knowledge_and_skills"]):
        questions.append(
            {
                "type": "prerequisites",
                "question": f"What prior knowledge and skills are expected for {course_ref}?",
                "answer_field": "content.prerequisite_knowledge_and_skills",
                "field_payload": course["prerequisite_knowledge_and_skills"],
                "course_code": code,
            }
        )

    if has_value(course["syllabus_of_seminars"]):
        questions.append(
            {
                "type": "seminars",
                "question": f"What is covered in seminars for {course_ref}?",
                "answer_field": "content.syllabus_of_seminars",
                "field_payload": course["syllabus_of_seminars"],
                "course_code": code,
            }
        )

    if has_value(course["language_of_instruction"]):
        questions.append(
            {
                "type": "language",
                "question": f"What is the language of instruction for {course_ref}?",
                "answer_field": "content.language_of_instruction",
                "field_payload": course["language_of_instruction"],
                "course_code": code,
            }
        )

    if has_value(course["completion"]):
        questions.append(
            {
                "type": "completion",
                "question": f"What academic criteria need to be met to obtain credits from {course_ref}?",
                "answer_field": "content.completion",
                "field_payload": course["completion"],
                "course_code": code,
            }
        )

    if has_value(course["time_span"]):
        questions.append(
            {
                "type": "time_span",
                "question": f"How is teaching time organized in {course_ref}?",
                "answer_field": "content.time_span",
                "field_payload": course["time_span"],
                "course_code": code,
            }
        )

    return questions


def generate_course_longform_single_field_set(
    input_dir: str,
    output_file: str,
    num_questions: int,
    seed: int,
    run_inference: bool,
    llm_config: LLMConfig,
) -> dict:
    rng = random.Random(seed)
    course_paths = sorted(Path(input_dir).glob("*.json"))
    courses = [course_record(path) for path in course_paths]

    all_candidates: list[dict] = []
    for c in courses:
        all_candidates.extend(candidate_questions_for_course(c, rng))

    by_type: dict[str, list[dict]] = {}
    for c in all_candidates:
        by_type.setdefault(c["type"], []).append(c)
    for t in by_type:
        rng.shuffle(by_type[t])

    # single-field only; balanced mix for long-form prompts
    quotas = {
        "learning_objectives": 10,
        "topics_yes": 8,
        "progress_assessment": 7,
        "study_literature": 6,
        "prerequisites": 5,
        "seminars": 4,
        "language": 4,
        "completion": 3,
        "time_span": 3,
    }

    sampled: list[dict] = []
    used_q = set()

    # pass 1: one-per-course preference within each type
    for t, q in quotas.items():
        pool = by_type.get(t, [])
        picked = 0
        used_courses = set()
        for item in pool:
            if item["question"] in used_q:
                continue
            if item["course_code"] in used_courses:
                continue
            sampled.append(item)
            used_q.add(item["question"])
            used_courses.add(item["course_code"])
            picked += 1
            if picked >= q:
                break

        if picked < q:
            for item in pool:
                if item["question"] in used_q:
                    continue
                sampled.append(item)
                used_q.add(item["question"])
                picked += 1
                if picked >= q:
                    break

    # pass 2: top-up if quotas were impossible for some types
    if len(sampled) < num_questions:
        pool = list(all_candidates)
        rng.shuffle(pool)
        for item in pool:
            if item["question"] in used_q:
                continue
            sampled.append(item)
            used_q.add(item["question"])
            if len(sampled) >= num_questions:
                break

    sampled = sampled[:num_questions]
    sampled = sorted(sampled, key=lambda x: (x["course_code"], x["type"], x["question"]))

    engine = VLLMInferenceEngine(llm_config) if run_inference else None

    items = []
    for i, item in enumerate(sampled, start=1):
        generated_answer = None
        if engine is not None:
            generated_answer = engine.short_answer(
                question=item["question"],
                field_name=item["answer_field"],
                field_text=item["field_payload"],
            )

        entry = {
            "id": f"courses_long_single_{i:03d}",
            "question": item["question"],
            "type": item["type"],
            "course_code": item["course_code"],
            "answer_field": item["answer_field"],
            "expected_answer": item.get("expected_answer") or item["field_payload"],
            "actual_answer": item.get("actual_answer") or item["field_payload"],
            "llm_short_answer": generated_answer,
        }
        items.append(entry)

    output = {
        "dataset": "FIT_courses_longform_single_field_v1",
        "num_questions": len(items),
        "seed": seed,
        "source": input_dir,
        "output_file": output_file,
        "run_inference": run_inference,
        "model": llm_config.model_name if run_inference else None,
        "description": "Long-form single-field course questions for LLM summarization; only answerable question-course pairs are included.",
        "items": items,
    }

    save_json(output, Path(output_file))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-form FIT benchmark questions using optional vLLM inference.")
    parser.add_argument("--target", choices=["courses-single-field"], default="courses-single-field")
    parser.add_argument("--input_dir", type=str, default="extracted_data_clean/fit/courses")
    parser.add_argument("--output_file", type=str, default="FIT_RAG_Benchmark/courses/longform_single_field.json")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run_inference", action="store_true", help="If set, run vLLM (Gemma 12B by default) to generate short answers.")
    parser.add_argument("--model_name", type=str, default="allenai/Olmo-3.1-32B-Instruct") #32B needs 8 24gbs with 2k max len and eager mode and 0.80 gpu mem util
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    llm_config = LLMConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    result = generate_course_longform_single_field_set(
        input_dir=args.input_dir,
        output_file=args.output_file,
        num_questions=args.num_questions,
        seed=args.seed,
        run_inference=args.run_inference,
        llm_config=llm_config,
    )

    print(f"Wrote {result['num_questions']} questions to {result['output_file']}")


if __name__ == "__main__":
    main()
