import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from unmute.llm.llm_utils import autoselect_model
from unmute.unmute_handler_speculative import SpeculativeState, UnmuteHandlerSpeculative


# Minimum RAG score for retrieved context to be considered usable.
# Below this, results are too weak to support a confident answer.
RAG_MIN_SCORE_THRESHOLD: float = 0.40

# Used when the question is complete AND RAG has good context.
_SPECULATIVE_MODE_SUFFIX = (
    "\n\n# SPECULATIVE GENERATION\n"
    "The user may still be speaking — their question may be incomplete.\n"
    "If the [RAG_CONTEXT] block clearly answers the question: give a concise 1-2 sentence answer now.\n"
    "If the context is missing, unclear, or the question seems unfinished: respond with ONLY a short filler phrase "
    "(e.g. \"Let me check that for you\", \"I believe it's\") then stop. The response will be continued.\n"
    "Do not fabricate. Do not write more than 2 sentences."
)

# Used when question is incomplete OR RAG context is too weak.
# Forces a deterministic filler — no LLM creativity, no hallucination risk.
_TEMPLATE_FILLER_SUFFIX = (
    "\n\n# INSTRUCTION\n"
    "The user's question is not yet complete or the context is insufficient to answer.\n"
    "Respond with ONLY the following phrase, word for word, nothing else:\n"
    "Let me check that for you."
)

RAG_CONTEXT_BLOCK_RE = re.compile(
    r"\n?\[RAG_CONTEXT\].*?\[/RAG_CONTEXT\]\s*",
    flags=re.DOTALL,
)


def strip_rag_context(text: str) -> str:
    return RAG_CONTEXT_BLOCK_RE.sub("", text).strip()


def collapse_rag_results(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in results:
        rank = row.get("rank")
        score = row.get("score")
        source_path = row.get("source_path")
        preview = str(row.get("preview", "")).strip()
        lines.append(f"[{rank}] score={score} source={source_path} :: {preview}")
    return "\n".join(lines)


def get_last_nonempty_user_index(chat_history: list[dict[str, Any]]) -> int | None:
    for idx in range(len(chat_history) - 1, -1, -1):
        row = chat_history[idx]
        if row.get("role") == "user" and str(row.get("content", "")).strip() != "":
            return idx
    return None


class RagRetriever:
    def __init__(
        self,
        rag_url: str,
        top_k: int,
        timeout_sec: float,
        max_retries: int = 1,
        retry_backoff_sec: float = 0.2,
    ):
        self.rag_url = rag_url.rstrip("/")
        self.top_k = top_k
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    async def retrieve(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": self.top_k,
        }

        url = f"{self.rag_url}/api/rag/retrieve"

        def _call_once() -> dict[str, Any]:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            decoded = json.loads(body)
            assert isinstance(decoded, dict)
            return decoded

        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(_call_once)
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_backoff_sec * (attempt + 1))

        raise RuntimeError("Unexpected RAG retry loop fallthrough")


class UnmuteHandlerSpeculativeRAG(UnmuteHandlerSpeculative):
    """
    Unmute speculative handler variant that introduces two RAG stages:
      1) speculative prefetch stage before speculative LLM starts,
      2) VAD-boundary refresh stage before generation commit/continuation.

    Rule: keep a single [RAG_CONTEXT] block by strip/replace on every stage.
    """

    def __init__(
        self,
        rag_url: str = "http://127.0.0.1:8095",
        rag_top_k: int = 5,
        rag_timeout_sec: float = 1.2,
    ) -> None:
        super().__init__()
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
        self._pending_spec_rag_collapsed: str = ""
        self._pending_spec_use_template: bool = False
        self._pending_vad_rag_task: asyncio.Task[Any] | None = None

    def get_rag_calls(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self._rag_calls]

    def _query_from_current_context(self) -> str:
        messages = self.chatbot.preprocessed_messages()
        user_joined = " ".join(
            str(m.get("content", ""))
            for m in messages
            if m.get("role") == "user"
        ).strip()
        query_source = self.chatbot.last_message("user") or user_joined
        return strip_rag_context(query_source)

    async def _check_question_complete(self, transcript: str) -> bool:
        """Return True if the transcript is a complete, specific, answerable question.

        Uses a tiny LLM call (max_tokens=5, temperature=0) for a YES/NO decision.
        Falls back to False (conservative — use template) on any error.
        Words fewer than 5 are always treated as incomplete without an LLM call.
        """
        if len(transcript.split()) < 5:
            return False
        messages = [
            {
                "role": "system",
                "content": (
                    "You decide if a speech transcript is a complete, specific question "
                    "that can be looked up or answered. Reply YES or NO only.\n"
                    "YES: grammatically complete, specific enough to retrieve an answer.\n"
                    "NO: cut off mid-sentence, too vague, or still being spoken."
                ),
            },
            {"role": "user", "content": f"Transcript: {transcript}"},
        ]
        try:
            resp = await self.openai_client.chat.completions.create(
                model=autoselect_model(),
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=5,
                stream=False,
            )
            text = (resp.choices[0].message.content or "").lower()
            return "yes" in text
        except Exception:
            return False

    def _apply_single_rag_context(self, collapsed: str) -> bool:
        idx = get_last_nonempty_user_index(self.chatbot.chat_history)
        if idx is None:
            return False
        current_user_text = str(self.chatbot.chat_history[idx].get("content", ""))
        base_user_text = strip_rag_context(current_user_text)
        self.chatbot.chat_history[idx]["content"] = (
            base_user_text.rstrip()
            + "\n\n[RAG_CONTEXT]\n"
            + collapsed
            + "\n[/RAG_CONTEXT]"
        )
        return True

    async def _run_rag_stage(
        self,
        *,
        stage: str,
        query: str,
        attempt_index: int | None = None,
        generation_index: int | None = None,
    ) -> dict[str, Any]:
        started_sec = self.audio_received_sec()
        started_wall = time.perf_counter()

        call: dict[str, Any] = {
            "stage": stage,
            "attempt_index": attempt_index,
            "generation_index": generation_index,
            "rag_started_sec": started_sec,
            "rag_finished_sec": None,
            "rag_input_query": query,
            "rag_top_k": self._rag_top_k,
            "rag_top_k_output": [],
            "rag_context_collapsed": None,
            "rag_error": None,
            "rag_latency_ms": None,
            "rag_applied_to_prompt": False,
            "rag_not_applied_reason": None,
        }

        if not query.strip():
            finished_sec = self.audio_received_sec()
            finished_wall = time.perf_counter()
            call["rag_finished_sec"] = finished_sec
            call["rag_latency_ms"] = (finished_wall - started_wall) * 1000.0
            call["rag_not_applied_reason"] = "empty_query"
        else:
            latest_user_before = strip_rag_context(self.chatbot.last_message("user") or "")
            try:
                rag_response = await self._retriever.retrieve(query)
                results_raw = rag_response.get("results", [])
                results = (
                    [x for x in results_raw if isinstance(x, dict)]
                    if isinstance(results_raw, list)
                    else []
                )
                topk_output = [
                    {
                        "rank": row.get("rank"),
                        "score": row.get("score"),
                        "chunk_id": row.get("chunk_id"),
                        "source_path": row.get("source_path"),
                        "preview": row.get("preview"),
                    }
                    for row in results[: self._rag_top_k]
                ]
                collapsed = collapse_rag_results(results[: self._rag_top_k])

                finished_sec = self.audio_received_sec()
                finished_wall = time.perf_counter()
                call["rag_finished_sec"] = finished_sec
                call["rag_top_k_output"] = topk_output
                call["rag_context_collapsed"] = collapsed
                call["rag_latency_ms"] = (finished_wall - started_wall) * 1000.0

                latest_user_after = strip_rag_context(self.chatbot.last_message("user") or "")
                if latest_user_after != latest_user_before:
                    call["rag_not_applied_reason"] = "user_resumed_or_changed_before_rag_ready"
                elif not collapsed:
                    call["rag_not_applied_reason"] = "empty_results"
                elif self._apply_single_rag_context(collapsed):
                    call["rag_applied_to_prompt"] = True
                else:
                    call["rag_not_applied_reason"] = "missing_user_message"
            except (urllib.error.URLError, TimeoutError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
                finished_sec = self.audio_received_sec()
                finished_wall = time.perf_counter()
                call["rag_finished_sec"] = finished_sec
                call["rag_error"] = repr(exc)
                call["rag_latency_ms"] = (finished_wall - started_wall) * 1000.0
                call["rag_not_applied_reason"] = "rag_error"

        self._rag_calls.append(call)

        if stage == "speculative_prefetch" and attempt_index is not None:
            self._update_spec_trace(
                attempt_index,
                rag_stage=stage,
                rag_started_sec=call["rag_started_sec"],
                rag_finished_sec=call["rag_finished_sec"],
                rag_input_query=call["rag_input_query"],
                rag_top_k_output=call["rag_top_k_output"],
                rag_context_collapsed=call["rag_context_collapsed"],
                rag_error=call["rag_error"],
                rag_latency_ms=call["rag_latency_ms"],
                rag_applied_to_prompt=call["rag_applied_to_prompt"],
                rag_not_applied_reason=call["rag_not_applied_reason"],
            )

        return call

    async def _pre_continuation_messages_hook(self) -> None:
        # If a VAD boundary RAG refresh is running concurrently (fired when spec
        # was committed), wait for it to inject into chat_history before the
        # continuation LLM takes its message snapshot.
        task = self._pending_vad_rag_task
        if task is not None:
            self._pending_vad_rag_task = None
            try:
                await task
            except Exception:
                pass

    def _spec_continuation_needed(self, committed_state: SpeculativeState) -> bool:
        # Spec always generates either a filler phrase ("Let me check") or a
        # short opener — never a complete standalone answer. Always continue so
        # the real VAD-triggered generation with fresh RAG finishes the response.
        return True

    def _get_speculative_messages(self) -> list[dict[str, Any]]:
        # Deep-copy only the entries we touch so we never mutate chat_history.
        messages = list(self.chatbot.preprocessed_messages())

        # Choose suffix based on whether we have a complete question + usable RAG.
        # Template mode → deterministic filler phrase, no LLM creativity.
        # Normal mode   → spec instructions + RAG context in system message.
        if self._pending_spec_use_template:
            suffix = _TEMPLATE_FILLER_SUFFIX
        else:
            suffix = _SPECULATIVE_MODE_SUFFIX
            rag = self._pending_spec_rag_collapsed
            if rag:
                suffix += f"\n\n# RETRIEVED CONTEXT\n{rag}"

        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + suffix

        # Strip any [RAG_CONTEXT] block that may already be in the user message
        # (injected by earlier speculative attempts) so the transcript stays clean.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                clean = strip_rag_context(str(messages[i].get("content", ""))).strip()
                if clean != messages[i].get("content"):
                    messages[i] = dict(messages[i])
                    messages[i]["content"] = clean
                break

        return messages

    async def _speculative_generation_task(self, state: SpeculativeState) -> None:
        query = self._query_from_current_context()

        # ── Stage 1: question completeness check ──────────────────────────────
        # Ask the LLM (tiny call, max_tokens=5) whether the partial transcript is
        # a complete, specific question. If not, skip RAG entirely and force a
        # deterministic template filler — no hallucination risk, no wasted RAG call.
        is_complete = await self._check_question_complete(query)

        if not is_complete:
            self._pending_spec_rag_collapsed = ""
            self._pending_spec_use_template = True
            try:
                await super()._speculative_generation_task(state)
            finally:
                self._pending_spec_use_template = False
                self._pending_spec_rag_collapsed = ""
            return

        # ── Stage 2: RAG + answerability ─────────────────────────────────────
        # Question is complete — retrieve context and decide whether to answer
        # or fall back to the template filler.
        call = await self._run_rag_stage(
            stage="speculative_prefetch",
            query=query,
            attempt_index=state.trace_index,
            generation_index=None,
        )

        collapsed = call.get("rag_context_collapsed") or ""
        top_result = (call.get("rag_top_k_output") or [{}])[0]
        top_score = float(top_result.get("score") or 0.0)
        rag_usable = (
            bool(collapsed)
            and not call.get("rag_error")
            and top_score >= RAG_MIN_SCORE_THRESHOLD
        )

        if rag_usable:
            self._pending_spec_rag_collapsed = collapsed
            self._pending_spec_use_template = False
            # Mark as applied via spec messages (not chat_history injection)
            if not call.get("rag_applied_to_prompt"):
                call["rag_applied_to_prompt"] = True
                call["rag_not_applied_reason"] = "injected_via_spec_messages"
                self._update_spec_trace(
                    state.trace_index,
                    rag_applied_to_prompt=True,
                    rag_not_applied_reason="injected_via_spec_messages",
                )
        else:
            self._pending_spec_rag_collapsed = ""
            self._pending_spec_use_template = True

        try:
            await super()._speculative_generation_task(state)
        finally:
            self._pending_spec_use_template = False
            self._pending_spec_rag_collapsed = ""

    async def _generate_response(self) -> None:
        # No user message yet → initial greeting; skip RAG and don't bump counter
        # so the counter stays aligned with the evaluation timing trace indices.
        if get_last_nonempty_user_index(self.chatbot.chat_history) is None:
            await super()._generate_response()
            return

        self._rag_generation_counter += 1
        query = self._query_from_current_context()

        # Quick check: do we have immediately-playable audio right now?
        spec = self._speculative
        primary_has_audio = (
            spec is not None and not spec.discarded and len(spec.audio_chunks) > 0
        )
        fallback_has_audio = (
            self._fallback_spec is not None and len(self._fallback_spec.audio_chunks) > 0
        )

        # Only wait for an in-flight graceful TTS drain if there is no audio
        # immediately available.  In offline eval, audio is fed faster than real-time
        # so the drain may not have finished by VAD even though wall-clock is close.
        # Skip the wait entirely when primary or fallback already has audio — adding
        # 1.5s latency there would hurt more than help.
        if not primary_has_audio and not fallback_has_audio:
            drain_task = self._graceful_drain_task
            if drain_task is not None and not drain_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(drain_task), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                # Recheck fallback after the wait
                fallback_has_audio = (
                    self._fallback_spec is not None and len(self._fallback_spec.audio_chunks) > 0
                )

        has_buffered_spec = primary_has_audio or fallback_has_audio

        if has_buffered_spec:
            # Fire the VAD boundary RAG refresh as a background task so the
            # committed speculative audio can drain immediately at VAD instead of
            # waiting ~350ms for the RAG HTTP call. _pre_continuation_messages_hook
            # will await this task before the continuation LLM reads messages.
            self._pending_vad_rag_task = asyncio.create_task(
                self._run_rag_stage(
                    stage="vad_boundary_refresh",
                    query=query,
                    attempt_index=None,
                    generation_index=self._rag_generation_counter,
                )
            )
            await super()._generate_response()
        else:
            # No committed spec — run RAG serially so it's in chat_history before
            # the normal LLM generation reads preprocessed_messages().
            await self._run_rag_stage(
                stage="vad_boundary_refresh",
                query=query,
                attempt_index=None,
                generation_index=self._rag_generation_counter,
            )
            await super()._generate_response()
