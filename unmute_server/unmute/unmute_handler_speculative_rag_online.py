"""
UnmuteHandlerSpeculativeRAGOnline

Online wrapper around UnmuteHandlerSpeculativeRAG that adds:
  - RagInstructions locked in __init__ (frontend cannot override)
  - update_session override (strips instructions, keeps voice only)
  - FAISS warmup in start_up() (avoids first-turn timeout)
  - compact RAG formatter via _run_rag_stage override
  - debug_dict["last_rag_call"] written for the frontend RAG panel

The offline handler (UnmuteHandlerSpeculativeRAG) is not modified.
"""
import logging
from typing import Any

import unmute.openai_realtime_api_events as ora
from unmute.llm.system_prompt import RagInstructions
from unmute.unmute_handler_rag import collapse_rag_results_compact, _clean_preview
from unmute.unmute_handler_speculative_rag import UnmuteHandlerSpeculativeRAG

logger = logging.getLogger(__name__)


class UnmuteHandlerSpeculativeRAGOnline(UnmuteHandlerSpeculativeRAG):
    """
    Online speculative RAG handler.

    Two RAG stages inherited from UnmuteHandlerSpeculativeRAG:
      1. speculative_prefetch — fires when anticipator p > 0.5 (~960 ms before VAD),
         queries on partial transcript, injects context before speculative LLM starts.
      2. vad_boundary_refresh — fires on real VAD, queries on complete transcript,
         overwrites context before continuation LLM runs.

    Online additions:
      - RagInstructions locked so frontend character picker cannot override.
      - Compact formatter: structural noise stripped, duplicates removed.
      - debug_dict["last_rag_call"] updated after each stage for the frontend panel.
      - FAISS warmup on start_up() to pre-load the index.
    """

    def __init__(
        self,
        rag_url: str = "http://127.0.0.1:8095",
        rag_top_k: int = 1,
        rag_timeout_sec: float = 1.2,
    ) -> None:
        super().__init__(
            rag_url=rag_url,
            rag_top_k=rag_top_k,
            rag_timeout_sec=rag_timeout_sec,
        )
        self.chatbot.set_instructions(RagInstructions())

    async def update_session(self, session) -> None:
        # Strip instructions sent by frontend — keep RagInstructions set in __init__.
        await super().update_session(
            ora.SessionConfig(
                voice=session.voice,
                allow_recording=session.allow_recording,
            )
        )

    async def start_up(self) -> None:
        await super().start_up()
        try:
            await self._retriever.retrieve("warmup")
            logger.info("[RAG] warmup retrieval done")
        except Exception as exc:
            logger.warning("[RAG] warmup retrieval failed: %r", exc)

    async def _run_rag_stage(
        self,
        *,
        stage: str,
        query: str,
        attempt_index: int | None = None,
        generation_index: int | None = None,
    ) -> dict[str, Any]:
        # Run parent retrieval + context injection
        call = await super()._run_rag_stage(
            stage=stage,
            query=query,
            attempt_index=attempt_index,
            generation_index=generation_index,
        )

        # Re-format with compact formatter (parent uses verbose collapse_rag_results)
        # and overwrite the injected context in chat_history.
        if call.get("rag_applied_to_prompt") and call.get("rag_top_k_output"):
            # Rebuild compact context from raw results
            raw_results = [
                {"preview": row.get("preview", "")}
                for row in call["rag_top_k_output"]
            ]
            compact = collapse_rag_results_compact(raw_results)
            if compact:
                from unmute.unmute_handler_rag import (
                    get_last_nonempty_user_index,
                    strip_rag_context,
                )
                idx = get_last_nonempty_user_index(self.chatbot.chat_history)
                if idx is not None:
                    base = strip_rag_context(
                        str(self.chatbot.chat_history[idx].get("content", ""))
                    )
                    self.chatbot.chat_history[idx]["content"] = (
                        base.rstrip()
                        + "\n\n[RAG_CONTEXT]\n"
                        + compact
                        + "\n[/RAG_CONTEXT]"
                    )
            call["rag_context_injected"] = compact
        else:
            call["rag_context_injected"] = ""

        # Expose to frontend RAG panel via debug_dict
        self.debug_dict["last_rag_call"] = {
            "generation_index": self._rag_generation_counter,
            "rag_input_query": call.get("rag_input_query", ""),
            "rag_context_injected": call.get("rag_context_injected", ""),
            "rag_applied_to_prompt": call.get("rag_applied_to_prompt", False),
            "rag_not_applied_reason": call.get("rag_not_applied_reason"),
            "rag_latency_ms": call.get("rag_latency_ms"),
            "rag_stage": stage,
        }

        return call
