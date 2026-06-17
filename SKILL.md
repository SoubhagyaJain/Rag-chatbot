---
name: production-rag
description: Use this skill whenever building or improving RAG systems that must be robust in production including local model deployments agentic integrations or high reliability requirements. Triggers include production RAG RAG architecture local RAG RAG evaluation chunking strategies hybrid retrieval reranking reliable RAG or grounded generation systems.
---

# Production-Ready Robust RAG Engineering

## Overview

This skill captures hard-won patterns from shipping multiple production-grade RAG systems, including fully local stacks and integrations into reliable agentic workflows. It treats RAG as a complete engineered system — not just retrieval plus generation. Focus is always on measurable reliability, latency/cost tradeoffs, failure mode coverage, and iterative improvement over raw model scale.

RAG is the right tool when you need grounding on private/dynamic/large corpora, attribution, or cost-efficient scaling beyond long-context limits. It fails in production primarily due to weak retrieval, poor chunking, missing fallbacks, and lack of rigorous evals — all of which this skill directly addresses.

## Core Principles (Apply These First)

- Retrieval quality dominates outcomes. 80% of gains come from better chunks, hybrid search, and reranking — not from swapping the generator model.
- Measure before optimizing. Define golden queries + expected answers/contexts early. Run automated evals on every change.
- Start simple, add complexity only when metrics justify it. Naive RAG baseline first, then layered improvements with A/B or before/after evals.
- Workflow engineering beats model power. Build explicit stages (classify → rewrite → retrieve → rerank → validate → fallback) with observability at each handoff.
- Local models shine for privacy, cost predictability, air-gapped deployments, and domain customization, but demand attention to quantization, batching, hardware profiling, and inference optimization.
- Never return ungrounded confident answers. Strong systems detect low-confidence retrieval or generation and escalate gracefully.

## Decision Framework: RAG vs Long-Context vs Agentic

- Static corpus under ~150-200k tokens and low update frequency → long-context (Claude 3.5/4 Sonnet/Opus or equivalent local long-context model) is often simpler and sufficient.
- Large, dynamic, private, or frequently updated data; need citations/attribution; cost or context-window pressure → RAG.
- Complex multi-hop reasoning, tool use beyond retrieval, or conversational state → agentic RAG (expose retriever as tool inside LangGraph/state machine or custom harness).
- Best of both: Use RAG for precise high-recall candidate selection, then feed top results + query into long-context model for synthesis. This pattern has delivered the highest faithfulness in systems I have shipped.

## Recommended 2026 Tech Stacks

**Production / Cloud Scale (managed services, high QPS):**
- Orchestration: LlamaIndex (best modular RAG components) or LangGraph (fine-grained agentic control).
- Embeddings: Voyage-3, OpenAI text-embedding-3-large, Cohere embed-v4, or domain-fine-tuned model.
- Vector store: Pinecone serverless, Weaviate, Qdrant, or PGVector on managed Postgres.
- Reranker: Cohere Rerank v3, Voyage rerank, or local bge-reranker + LLM rerank fallback.
- Generator: Claude 3.5 Sonnet / 4 family (superior grounding and instruction following) or GPT-4o / Grok. Route easy queries to cheaper/faster models.
- Observability: LangSmith, Arize Phoenix, Helicone, or OpenTelemetry + custom dashboards.

**Local / On-Prem / Privacy-First / Cost-Optimized (what I have personally productionized):**
- LLM serving: Ollama (fastest to start, great for dev), vLLM or Text Generation Inference (TGI) for high-throughput production, llama.cpp or MLC LLM for edge/CPU.
- Embeddings: fastembed (Rust backend, excellent speed/quality), sentence-transformers with ONNX optimization, nomic-embed-text, BGE-large-en-v1.5, or UAE-Large-V1.
- Vector store: Chroma (simplest local), LanceDB (strong performance + embedded), FAISS (maximum control, manual sharding), or local PGVector.
- Orchestration: LlamaIndex configured for local models + custom IngestionPipeline; or lightweight custom Python with Pydantic + asyncio.
- Serving layer: FastAPI (async endpoints, streaming), Dockerized, optional Kubernetes with HPA. Add semantic caching (Redis or in-memory with embedding similarity).
- Quantization: GGUF Q4_K_M / Q5_K_M for balanced quality/speed; AWQ or GPTQ for GPU; bitsandbytes for flexible HF loading. Profile VRAM/RAM carefully (70B Q4 ≈ 40 GB, 32B Q4 ≈ 20 GB, 8-14B Q5 often fits single consumer GPU).

Local stack example that has worked well for me on 100k+ chunk corpora: Ollama (Qwen2.5-32B or Llama-3.3-70B quantized) + fastembed (BGE) + Chroma + LlamaIndex query engine + FastAPI with streaming + semantic cache. Achieved sub-2s p95 end-to-end on modest hardware with proper caching and query routing.

## Chunking — The Highest-ROI Decision

Poor chunking is the #1 silent killer of RAG quality.

- Baseline: RecursiveCharacterTextSplitter with chunk_size 800-1200 tokens, chunk_overlap 100-200 tokens. Preserve structure (headers, lists, tables via markdown).
- Next level (strongly recommended): Semantic chunking — embed sentences or small windows and split where embedding similarity drops. LlamaIndex SemanticSplitterNodeParser or custom implementation using cosine threshold.
- For long/complex documents: Hierarchical indexing + auto-merging retriever (small child chunks + parent summary chunks). LlamaIndex has excellent support.
- Domain adaptations:
  - Code: split by functions, classes, or logical blocks; preserve signatures and docstrings.
  - Tabular/PDF: use unstructured or marker for layout-aware extraction; represent tables as markdown or JSON.
  - Legal/Regulatory: split by section/article; attach metadata for effective date, jurisdiction.
- Always enrich nodes with metadata: source file, page/section, last_updated, extracted entities (NER), short summary, keywords. This enables powerful pre-filtering and improves reranker signals.
- Target chunk size: 400-1500 tokens depending on embedding model and generator context. Test retrieval metrics on your golden set.

Use LlamaIndex NodeParser / IngestionPipeline for reproducibility and easy metadata attachment.

## Robust Multi-Stage Retrieval Pipeline

Implement retrieval as an explicit pipeline, not a single vector search call.

1. **Query Classification & Preprocessing** (use small/fast model or rules):
   - Route factual/simple queries to cached or lightweight path.
   - Detect multi-hop, analytical, or conversational intent → adjust retrieval strategy or number of candidates.
   - Extract structured filters (dates, entities, document types) for metadata pre-filtering.
   - Query rewriting: HyDE, multi-query generation, or step-back prompting. Run in parallel with small model.

2. **First-Stage Retrieval** (over-retrieve):
   - Hybrid search: dense vector (cosine/IP) + sparse (BM25 or SPLADE) + metadata filters. Most modern vector DBs support this natively.
   - Retrieve top 20-50 candidates. Diversity via MMR if redundancy is high.

3. **Post-Retrieval Optimization** (critical for precision):
   - Reranking: Cross-encoder (bge-reranker-large local is strong and fast) or API (Cohere/Voyage). For maximum quality, use Claude or strong local LLM to rerank with a structured prompt ("Rank the following chunks by relevance... output JSON with scores").
   - Contextual compression or sentence-window retrieval.
   - Relevance filtering: drop chunks below a dynamic threshold (calibrated on golden set).

4. **Context Assembly & Prompting**:
   - Select top 5-10 chunks (or fewer with long-context generator).
   - Structure prompt clearly: system instructions emphasizing "Use ONLY the provided context. Quote verbatim when possible. If information is insufficient, explicitly state you do not have enough information."
   - Include citations or source metadata in response when possible.
   - Anthropic-style prompting: Use XML-inspired tags or clear delimiters for context blocks.

## Generation, Validation & Grounding

- Strict grounding prompt + few-shot examples of good vs bad behavior.
- Post-generation validation layer (LLM-as-judge or entailment model): "Does every factual claim in the answer appear in or logically follow from the provided context? Output yes/no + explanation." If no, trigger self-critique loop (max 2 iterations) or fallback.
- For weaker local models: more explicit instructions, structured output (JSON mode or constrained decoding), and stronger validation.
- Streaming + source citation improves UX and trust.

## Evaluation — Build This Before the Pipeline

You cannot improve what you do not measure.

1. Create a golden evaluation set: 50–200 realistic query–answer–supporting-context triples. Source from production logs (anonymized), domain experts, or high-quality synthetic generation with Claude followed by human review.
2. Metrics:
   - Retrieval: context_precision, context_recall, hit_rate@k, NDCG, MRR (RAGAS or custom).
   - Generation: faithfulness (hallucination rate via LLM judge or NLI), answer_relevancy, answer_correctness (semantic + factual), citation accuracy.
   - System: end-to-end task success rate on golden set, p50/p95 latency, cost per query, fallback rate, user thumbs up/down.
3. Tools: RAGAS (excellent LlamaIndex integration), DeepEval, Arize Phoenix (tracing + eval), or custom harness using Claude as judge (very reliable for faithfulness and correctness).
4. Production monitoring: Track retrieval score distributions over time (drift detection), failure mode taxonomy from logs, and online A/B or shadow testing of new retrieval strategies.

Run the full eval suite on every significant change. Treat eval score regression as a blocking CI signal.

## Reliability Patterns That Separate Good from Production-Grade Systems

- **Graceful degradation & fallbacks**: Low retrieval confidence or validation failure → "I don't have sufficient information in the available documents to answer reliably." or route to stronger model / human escalation queue. Never hallucinate confidently.
- **Semantic + exact caching**: Cache (query_embedding, answer) pairs. On new query, check similarity to past successful queries. Invalidate on index updates. Huge latency and cost wins.
- **Query routing & model cascade**: Easy queries → small/fast/cheap model or cache. Hard queries → full RAG + strong generator. Multi-hop or tool-needed → agent loop.
- **Self-correction loops**: Generation → validation → (if fail) rewrite query or retrieve more → regenerate. Limit iterations.
- **Observability everywhere**: Trace query_id through every stage (rewrite, retrieved_ids + scores, reranked_ids, prompt, generation, validation_result, fallback_triggered). Essential for debugging why a specific query failed.
- **Incremental & versioned indexing**: Detect document changes (webhooks, S3 events, DB CDC). Re-process only changed chunks. Maintain index versions for rollback.
- **Security & compliance**: PII redaction before indexing, metadata-based access control (row-level security), audit logging of retrieved sources per query. Local deployments can be fully air-gapped.

## Local Model Specific Engineering (Hands-On Lessons)

- **Embedding choice matters hugely** on domain data. Test 2-3 candidates on your golden retrieval metrics. fastembed makes iteration fast.
- **Quantization strategy**: Start Q5_K_M or Q4_K_M. Measure quality drop on your eval set. Some domains tolerate Q3 better than others.
- **Inference optimization**: vLLM continuous batching + paged attention for throughput. TensorRT-LLM or ONNX Runtime for maximum speed on specific hardware. For CPU-heavy: llama.cpp with proper thread/ batch tuning.
- **Memory & context management**: Use activation checkpointing / gradient checkpointing concepts if fine-tuning; for inference, manage KV cache carefully. For very long local context, combine with RAG to keep effective context focused.
- **Offline/air-gapped**: Pre-download all models and embedding caches. Bundle everything in Docker images or volumes. Test cold-start and model loading times.
- **Cost/latency reality check**: Local 70B Q4 on good GPU can be faster and cheaper than API for high volume once amortized, but requires ops investment. Profile end-to-end before committing.

## Agentic RAG & Integration into Reliable Workflows

Expose RAG as a well-defined tool with input schema (query, optional_filters, top_k, include_sources). 

Inside a LangGraph or custom state machine:
- Planner decides whether retrieval is needed and with what strategy.
- Retriever tool executes the multi-stage pipeline above.
- Synthesizer (strong model) produces answer + citations.
- Critic / Validator checks faithfulness and completeness.
- If confidence low or missing info: Clarify with user, escalate to human, or fall back to broader search / different tool.

Add memory: conversation history in prompt; long-term memory of past high-quality retrievals or user corrections that can be retrieved in future similar queries.

This "harness + validation + escalation" pattern is what turns brittle RAG demos into systems that actually work reliably in production.

## Implementation Roadmap (Follow in Order)

1. Define success metrics and build/evaluate golden dataset.
2. Ingest → chunk (with metadata) → embed → index. Use LlamaIndex IngestionPipeline for reproducibility.
3. Implement baseline naive retriever + generator. Run full eval suite. Record scores.
4. Add query classification/rewriting + hybrid search. Re-eval and compare.
5. Add reranker + contextual compression. Re-eval.
6. Add caching layer, post-generation validation, and explicit fallbacks. Measure reliability and latency impact.
7. Instrument full tracing and monitoring. Set up dashboards and alerts on key metrics (fallback rate, faithfulness score distribution).
8. Containerize (Docker Compose for local, K8s manifests for prod). Add health checks, resource limits, model pre-loading.
9. Deploy shadow traffic or canary. Collect real failure cases and expand golden set.
10. Establish continuous improvement loop: weekly review of low-performing queries, targeted chunking or prompt improvements, periodic re-indexing or embedder fine-tuning.

## Common Production Failure Modes & Proven Fixes

- Relevant information split across chunks or lost in noise → semantic chunking + hierarchical indexing.
- Domain mismatch with generic embeddings → test and switch to stronger domain-adapted or fine-tuned embedder.
- Too much noise in top-k → mandatory reranking stage + relevance thresholding.
- Model ignores context or hallucinates anyway → stricter grounding prompts, few-shot examples, post-generation validation loop.
- No graceful handling of insufficient context → explicit "I don't have enough information" path + escalation.
- Eval scores good but production bad (distribution shift) → continuous sampling of real queries into eval set + online monitoring.
- Latency or cost explodes at scale → query routing, semantic caching, smaller models for easy cases, and async pipeline stages.
- Local model quality collapses under load → proper quantization profiling, batching, and fallback to stronger model or API for hard queries.

## Next Steps & Resources

After internalizing this skill, you should be able to design, implement, evaluate, and productionize a robust RAG system — local or cloud — with clear reliability targets and a path to continuous improvement.

For concrete code templates, Docker Compose examples for local stacks, detailed prompt templates, sample LlamaIndex configurations, eval harness scripts, and architecture decision records, see the `references/` directory in this skill.

Iterate on your implementation, run the evals religiously, and treat every production failure as a signal to strengthen the workflow harness. That is how reliable RAG systems are built.

---
*Skill authored from the perspective of deep production experience shipping grounded retrieval systems at scale, including fully local deployments and integration into agentic platforms.*
