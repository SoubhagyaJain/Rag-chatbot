"""
Chainlit chat interface for the Company Policy RAG agent.

Run:
    chainlit run app/chat_app.py --port 8000

Citation UX (production-rag principle: trust through verifiable sources):
- Answer appears in the main message bubble.
- Sources render as expandable Chainlit Text elements below the answer.
- Each source shows section_path + page + excerpt so users can verify policy text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when Chainlit loads this module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from llama_index.core.agent import ReActAgent

from src.agent import AgentTurnResult, chat_with_memory, get_agent_context
from src.memory import memory_stats
from src.config import settings
from src.indexing import get_collection_stats, index_exists
from src.utils import (
    build_chainlit_citation_elements,
    logger,
    prepare_citations_for_display,
    setup_logging,
)

setup_logging("chainlit")


def _sources_header(count: int) -> str:
    """Compact header above expandable source elements."""
    if count == 0:
        return ""
    noun = "source" if count == 1 else "sources"
    return f"---\n📚 **{count} {noun}** — expand any item below to verify against the policy document."


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize agent and retriever once per user session."""
    if not index_exists():
        await cl.Message(
            content=(
                "⚠️ **No document index found.**\n\n"
                "Please run indexing first:\n"
                "```bash\n"
                "python scripts/index_documents.py\n"
                "```\n"
                "Place PDFs in `data/policies/` and `data/legal/` before indexing."
            )
        ).send()
        return

    try:
        ctx = get_agent_context()
        cl.user_session.set("agent", ctx["agent"])
        cl.user_session.set("retriever", ctx["retriever"])
        cl.user_session.set("memory", ctx["memory"])
    except Exception as exc:
        logger.exception("Failed to initialize agent")
        await cl.Message(content=f"Failed to start agent: {exc}").send()
        return

    citations_on = "enabled" if settings.show_citations else "disabled"
    chroma_stats = get_collection_stats()
    if settings.enable_reranker:
        retrieval_line = (
            f"- Retrieval: `{settings.retrieval_candidate_k}` candidates → "
            f"rerank → top `{settings.reranker_top_n}` (`{settings.reranker_model}`)\n"
        )
    else:
        retrieval_line = f"- Top-K retrieval: `{settings.similarity_top_k}` (reranker off)\n"
    mem_stats = memory_stats(cl.user_session.get("memory"))
    if mem_stats["enabled"]:
        memory_line = (
            f"- Conversation memory: **on** "
            f"(last `{settings.memory_window_size}` turns, "
            f"{settings.memory_token_limit} token cap)\n"
        )
    else:
        memory_line = "- Conversation memory: **off**\n"
    await cl.Message(
        content=(
            f"**Company Policy RAG** is ready.\n\n"
            f"- LLM: `{settings.llm_model}`\n"
            f"- Embeddings: `{settings.embed_model}`\n"
            f"- Vector store: **ChromaDB** (`{chroma_stats['collection']}`)\n"
            f"- Indexed chunks: **{chroma_stats['count']}**\n"
            f"{retrieval_line}"
            f"{memory_line}"
            f"- Source citations: **{citations_on}**\n\n"
            "Ask about employee policies, handbook rules, contracts, or legal documents. "
            "You can ask follow-up questions — the assistant remembers recent turns in this chat."
        )
    ).send()


async def _send_answer_with_citations(
    answer: str,
    citations: list[dict],
) -> None:
    """
    Send the agent answer with an attached Sources section.

    Sources are Chainlit Text elements (expandable/collapsible) rather than a
    second message — keeps the conversation thread readable and groups answer
    with its evidence. Future: attach file_path for PDF deep-linking.
    """
    elements: list = []
    footer = ""

    if settings.show_citations and citations:
        prepared = prepare_citations_for_display(citations)
        if prepared:
            elements = build_chainlit_citation_elements(citations)
            footer = _sources_header(len(prepared))

    content = answer
    if footer:
        content = f"{answer}\n\n{footer}"

    await cl.Message(content=content, elements=elements).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle user message via ReAct agent; attach rich source citations."""
    agent: ReActAgent | None = cl.user_session.get("agent")
    retriever = cl.user_session.get("retriever")
    memory = cl.user_session.get("memory")

    if agent is None:
        await cl.Message(
            content="Agent not initialized. Please refresh after running the indexing script."
        ).send()
        return

    user_text = message.content.strip()
    if not user_text:
        return

    async with cl.Step(name="Searching policies", type="tool") as step:
        step.input = user_text
        try:
            turn: AgentTurnResult = await chat_with_memory(
                agent, user_text, memory=memory
            )
            step.output = turn.answer[:500] + (
                "..." if len(turn.answer) > 500 else ""
            )
        except Exception as exc:
            logger.exception("Agent query failed")
            await cl.Message(content=f"Sorry, something went wrong: {exc}").send()
            return

    await _send_answer_with_citations(turn.answer, turn.citations)


@cl.on_chat_end
async def on_chat_end() -> None:
    logger.info("Chat session ended")


if __name__ == "__main__":
    import subprocess

    subprocess.run(
        ["chainlit", "run", str(Path(__file__).resolve()), "--port", str(settings.chainlit_port)],
        check=False,
    )