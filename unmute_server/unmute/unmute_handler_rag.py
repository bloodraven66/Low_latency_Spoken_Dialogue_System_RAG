import logging
import re
import time
import urllib.error
from typing import Any

from unmute.llm.system_prompt import RagInstructions
from unmute.unmute_handler import UnmuteHandler
from unmute.unmute_handler_speculative_rag import (
    RagRetriever,
    get_last_nonempty_user_index,
    strip_rag_context,
)

logger = logging.getLogger(__name__)

_FIELD_RE = re.compile(r"\[FIELD=([^\]]+)\]")
_DISCARD_KEYS = {"entity_type", "entity_id"}


def _clean_preview(preview: str) -> str:
    """Strip structural noise from a preview, keep all meaningful content."""
    # Split header (before [FIELD=...]) from the field value body (after it)
    field_match = _FIELD_RE.search(preview)
    if field_match:
        header = preview[: field_match.start()]
        field_key = field_match.group(1)          # e.g. "contact.details.e_mail"
        field_value = preview[field_match.end() :].strip()
    else:
        header = preview
        field_key = None
        field_value = None

    # Strip "[CONTEXT]" prefix from header
    header = re.sub(r"^\[CONTEXT\]\s*", "", header.strip())

    # Split header into key=value parts, drop unwanted keys
    kept_parts = []
    for part in header.split("|"):
        part = part.strip()
        if not part:
            continue
        key = part.split("=", 1)[0].strip()
        if key in _DISCARD_KEYS:
            continue
        kept_parts.append(part)

    header_clean = " | ".join(kept_parts)

    # Append the specific matched field value only if it adds new info
    # (field_value is often already captured as one of the header key=value pairs)
    if field_key and field_value:
        short_key = field_key.split(".")[-1]
        # Check if the value is already present in the cleaned header
        if field_value not in header_clean:
            header_clean += f" | {short_key}: {field_value}"

    return header_clean


def collapse_rag_results_compact(results: list[dict[str, Any]]) -> str:
    """Compact formatter — no score/source, structural RAG noise stripped, duplicates removed."""
    lines: list[str] = []
    seen: set[str] = set()
    rank = 1
    for row in results:
        preview = str(row.get("preview", "")).strip()
        cleaned = _clean_preview(preview)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        lines.append(f"[{rank}] {cleaned}")
        rank += 1
    return "\n".join(lines)


class UnmuteHandlerRAG(UnmuteHandler):
    """
    Non-speculative RAG handler. At VAD pause boundary, retrieves context from
    the RAG service and injects it into the last user message before LLM generation.
    Falls back to normal generation on any RAG error or empty result.
    """

    def __init__(
        self,
        rag_url: str = "http://127.0.0.1:8095",
        rag_top_k: int = 3,
        rag_timeout_sec: float = 1.2,
    ) -> None:
        super().__init__()
        self.chatbot.set_instructions(RagInstructions())
        self._retriever = RagRetriever(
            rag_url=rag_url,
            top_k=rag_top_k,
            timeout_sec=rag_timeout_sec,
            max_retries=1,
            retry_backoff_sec=0.25,
        )
        self._rag_top_k = rag_top_k
        self._rag_calls: list[dict[str, Any]] = []
        self._rag_generation_counter: int = -1

    async def update_session(self, session) -> None:
        # Always keep RAG instructions regardless of what the frontend sends.
        # Voice changes are still applied normally.
        import unmute.openai_realtime_api_events as ora
        await super().update_session(ora.SessionConfig(
            voice=session.voice,
            allow_recording=session.allow_recording,
        ))

    def get_rag_calls(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._rag_calls]

    async def start_up(self) -> None:
        await super().start_up()
        try:
            await self._retriever.retrieve("warmup")
            logger.info("[RAG] warmup retrieval done")
        except Exception as exc:
            logger.warning("[RAG] warmup retrieval failed: %r", exc)

    def _query_from_current_context(self) -> str:
        last_msg = self.chatbot.last_message("user") or ""
        return strip_rag_context(last_msg)

    def _apply_rag_context(self, collapsed: str) -> bool:
        idx = get_last_nonempty_user_index(self.chatbot.chat_history)
        if idx is None:
            return False
        current = str(self.chatbot.chat_history[idx].get("content", ""))
        base = strip_rag_context(current)
        self.chatbot.chat_history[idx]["content"] = (
            base.rstrip() + "\n\n[RAG_CONTEXT]\n" + collapsed + "\n[/RAG_CONTEXT]"
        )
        return True

    async def _generate_response(self) -> None:
        self._rag_generation_counter += 1
        generation_index = self._rag_generation_counter

        query = self._query_from_current_context()
        started_wall = time.perf_counter()

        call: dict[str, Any] = {
            "generation_index": generation_index,
            "rag_started_sec": None,
            "rag_finished_sec": None,
            "rag_latency_ms": None,
            "rag_input_query": query,
            "rag_top_k": self._rag_top_k,
            "rag_top_k_output": [],
            "rag_context_injected": "",
            "rag_applied_to_prompt": False,
            "rag_not_applied_reason": None,
            "rag_error": None,
        }

        if not query.strip():
            call["rag_latency_ms"] = (time.perf_counter() - started_wall) * 1000.0
            call["rag_not_applied_reason"] = "empty_query"
            logger.info("[RAG] gen=%d skipped: empty query", generation_index)
        else:
            logger.info("[RAG] gen=%d query: %r", generation_index, query)
            user_before = strip_rag_context(self.chatbot.last_message("user") or "")
            try:
                rag_response = await self._retriever.retrieve(query)
                results_raw = rag_response.get("results", [])
                results = [x for x in results_raw if isinstance(x, dict)] if isinstance(results_raw, list) else []
                topk = results[: self._rag_top_k]
                collapsed = collapse_rag_results_compact(topk)

                call["rag_latency_ms"] = (time.perf_counter() - started_wall) * 1000.0
                call["rag_top_k_output"] = [
                    {k: row.get(k) for k in ("rank", "score", "chunk_id", "source_path", "preview")}
                    for row in topk
                ]

                logger.info(
                    "[RAG] gen=%d latency=%.0f ms results=%d",
                    generation_index, call["rag_latency_ms"], len(topk),
                )
                logger.info("[RAG] gen=%d --- RAW results ---", generation_index)
                for r in topk:
                    logger.info(
                        "[RAG]   [%s] score=%.3f source=%s\n        preview: %s",
                        r.get("rank"), r.get("score"), r.get("source_path"),
                        str(r.get("preview", "")),
                    )
                logger.info("[RAG] gen=%d --- LLM context ---\n%s", generation_index, collapsed)

                user_after = strip_rag_context(self.chatbot.last_message("user") or "")
                if user_after != user_before:
                    call["rag_not_applied_reason"] = "user_changed_before_rag_ready"
                    logger.info("[RAG] gen=%d not applied: user changed before RAG ready", generation_index)
                elif not collapsed:
                    call["rag_not_applied_reason"] = "empty_results"
                    logger.info("[RAG] gen=%d not applied: empty results", generation_index)
                elif self._apply_rag_context(collapsed):
                    call["rag_context_injected"] = collapsed
                    call["rag_applied_to_prompt"] = True
                    logger.info("[RAG] gen=%d applied to prompt", generation_index)
                else:
                    call["rag_not_applied_reason"] = "missing_user_message"
                    logger.info("[RAG] gen=%d not applied: missing user message", generation_index)

            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                call["rag_latency_ms"] = (time.perf_counter() - started_wall) * 1000.0
                call["rag_error"] = repr(exc)
                call["rag_not_applied_reason"] = "rag_error"
                logger.warning("[RAG] gen=%d error: %r", generation_index, exc)

        self._rag_calls.append(call)
        self.debug_dict["last_rag_call"] = call

        messages = self.chatbot.preprocessed_messages()
        logger.info("[RAG] gen=%d full prompt (%d messages):", generation_index, len(messages))
        for msg in messages:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))
            logger.info("[RAG]   [%s] %s", role, content)

        await super()._generate_response()
