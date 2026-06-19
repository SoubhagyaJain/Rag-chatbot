"""SSE streaming chat: retrieve then stream synthesis tokens."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from llama_index.core import Settings

import uuid

from src.citations import (
    begin_citation_turn,
    get_generation_nodes_this_turn,
    record_generation_sources,
    select_citations_for_answer,
)
from src.config import settings as app_settings
from src.generation import (
    _format_text_chunks,
    apply_faithfulness_guard,
    build_grounded_response_synthesizer,
    normalize_balanced_answer,
)
from src.code_validation import apply_code_validation_pipeline
from src.language import append_language_hint
from src.prompts import format_nodes_for_prompt, get_text_qa_template
from src.retrieval_trace import build_retrieval_trace
from src.thinking_extract import (
    begin_thinking_turn,
    get_thinking_this_turn,
    set_thinking_this_turn,
    split_llm_response,
)
from src.timing import begin_query_timing, clear_timing, get_current_timing, record_stage


def _sse(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_context(nodes: list) -> str:
    parts = [format_nodes_for_prompt([n]) for n in nodes]
    return "\n\n".join(parts)


def stream_chat_turn(
    message: str,
    nodes: list,
    *,
    t0: float,
) -> Iterator[str]:
    """
    Yield SSE events after retrieval is complete.

    Events: retrieval_done, thinking, token, done, error
    """
    try:
        record_generation_sources(nodes)
        timing = get_current_timing()
        yield _sse(
            "retrieval_done",
            build_retrieval_trace(nodes, timing.as_dict() if timing else None),
        )

        synth = build_grounded_response_synthesizer()
        llm = synth._llm or Settings.llm
        query_str = append_language_hint(message)
        context = _build_context(nodes)
        template = get_text_qa_template()
        prompt = template.format(context_str=context, query_str=query_str)

        raw_accumulated = ""
        streamed = False

        if hasattr(llm, "stream_complete"):
            try:
                for chunk in llm.stream_complete(prompt):
                    delta = getattr(chunk, "delta", None) or str(chunk)
                    if not delta:
                        continue
                    raw_accumulated += delta
                    streamed = True
                    if "<think>" in raw_accumulated.lower() and "</think>" not in raw_accumulated.lower():
                        continue
                    split_partial = split_llm_response(
                        raw_accumulated, model_id=app_settings.llm_model
                    )
                    if split_partial.thinking and not get_thinking_this_turn():
                        set_thinking_this_turn(split_partial.thinking)
                        yield _sse("thinking", split_partial.thinking)
                    visible = split_partial.answer
                    if visible:
                        yield _sse("token", delta)
            except Exception:
                streamed = False

        if not streamed:
            text_chunks = _format_text_chunks(nodes)
            raw_accumulated = synth.get_response(
                query_str=query_str,
                text_chunks=text_chunks,
            )

        model_id = app_settings.llm_model
        split = split_llm_response(raw_accumulated, model_id=model_id)
        if split.thinking:
            set_thinking_this_turn(split.thinking)
            yield _sse("thinking", split.thinking)

        answer = normalize_balanced_answer(split.answer, query=message)
        answer = apply_faithfulness_guard(answer, nodes, llm)
        answer, _ = apply_code_validation_pipeline(message, answer, nodes, llm)

        if not streamed:
            for word in answer.split():
                yield _sse("token", word + " ")

        record_stage("e2e", (time.perf_counter() - t0) * 1000)
        timing = get_current_timing()
        generation_nodes = get_generation_nodes_this_turn()
        citations: list[dict[str, Any]] = []
        if app_settings.show_citations:
            citations = select_citations_for_answer(
                answer,
                generation_nodes,
                user_query=message,
            )
        yield _sse(
            "done",
            {
                "answer": answer,
                "thinking": get_thinking_this_turn(),
                "timing": timing.as_dict() if timing else None,
                "citations": citations,
                "retrieval_trace": build_retrieval_trace(
                    generation_nodes,
                    timing.as_dict() if timing else None,
                ),
                "message_id": str(uuid.uuid4())[:12],
                "low_confidence": "low confidence" in answer.lower(),
                "grounding_mode": app_settings.grounding_strictness,
            },
        )
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
    finally:
        clear_timing()


def prepare_stream_turn(message: str) -> tuple[float, None]:
    begin_query_timing()
    begin_citation_turn()
    begin_thinking_turn()
    return time.perf_counter(), None