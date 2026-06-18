"""
Follow-up answer bank for interview-notes.html.

Maps (parent_question_title, followup_question) from _gen_interview_notes.py
RAW_QUESTIONS to HTML answer strings consumed by the notes generator.
"""
from __future__ import annotations

import html


def para(text: str) -> str:
    """Escape *text* and wrap each double-newline-separated block in <p>...</p>."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(p)}</p>" for p in parts)


ANSWERS: dict[tuple[str, str], str] = {
    ('Walk me through this project in 60 seconds.', 'What would you do differently?'): para(
        'I would add hybrid BM25 for exact legal clause matches and wire an independent Claude-as-judge for faithfulness scoring in evaluation.py, while keeping local qwen2.5:7b generation. I would also add a CI eval smoke gate on the 15 golden cases so no change merges if relevancy drops below 0.70. At 81 chunks the retrieval stack is solid; those upgrades close the biggest gaps before multi-doc scale.'
    ),
    ('Walk me through this project in 60 seconds.', 'Biggest failure?'): para(
        'Run 091001: answer relevancy collapsed to 0.40 even though context precision held at 0.82—the faithfulness guard in generation.py was replacing substantive answers with abstention boilerplate. I traced it via guard_modified flags in evaluation.py and fixed prompts, few-shots, and guard behavior across four phases. Recovery landed at 0.747 relevancy on run 104356 without hiding the regression in logs.'
    ),
    ("Why is this relevant to Anthropic's work?", 'How do you calibrate judges?'): para(
        'The golden set in golden_dataset.json has rubric expected_answer fields per case—I use those as anchor criteria. The relevancy judge in evaluation.py sees retrieved context plus the question so it does not penalize correct abstention when the topic is absent. I log per-case judge_notes and compare trends across 13 eval runs rather than trusting a single aggregate score.'
    ),
    ("Why is this relevant to Anthropic's work?", 'What about scalable oversight?'): para(
        'At 15 cases I do human-style rubric design with LLM execution; the scalable piece is append-only evaluation_results.json that catches regressions like 091001 automatically. For growth I would stratify golden cases by failure mode (faithfulness, relevancy, citation precision), add human spot-checks on disagreements, and use strict vs balanced modes in prompts.py as explicit oversight knobs for auditors vs employees.'
    ),
    ('What was your role and scope?', 'How long did it take?'): para(
        'Git shows one commit on 2026-06-17 (~0.5 calendar weeks in span); the real work was an intensive same-day solo sprint—13 logged eval runs over ~5.4 hours UTC (05:22–10:44), best balanced run 104356. Scope: indexing through Streamlit UI, Docker, 94 pytest tests, LlamaIndex 0.14 migration. Each full 15-case eval takes ~16 minutes on CPU; the harness in evaluation.py was the forcing function—changes did not ship without a trend line. Source: docs/project_timeline.json (git log + evaluation_results.json).'
    ),
    ('What was your role and scope?', 'What would a team split look like?'): para(
        'I would split retrieval/indexing (retriever.py, indexing.py, query_processing.py), generation/safety (generation.py, prompts.py, citations.py), and platform/eval (evaluation.py, agent.py, Streamlit, Docker). config.py stays shared so retrieval and eval never drift. As solo builder I wore all three hats, which is why get_retrieval_config_summary() snapshots settings per eval run.'
    ),
    ('Tell me about a time a metric regressed.', 'How did you detect it?'): para(
        'Append-only evaluation_results.json showed run 091001: relevancy 0.40 vs 0.71 baseline while context precision stayed ~0.82—retrieval was not the culprit. Per-case traces with guard_modified=True pointed to apply_faithfulness_guard in generation.py replacing good answers. I reproduced with a single golden case before re-running all 15.'
    ),
    ('Tell me about a time a metric regressed.', 'What guardrails prevent recurrence?'): para(
        'Every eval run logs config snapshots, guard_modified flags, and pre_guard_answer traces. I added pytest coverage for guard behavior, citation selection, and normalize_balanced_answer. Before declaring victory I now check faithfulness and relevancy together—strict mode taught me that faith 1.00 with relevancy 0.42 is a silent failure.'
    ),
    ('What would you put on a Claude system card for this?', 'Residual risk?'): para(
        'Residual risks: faithfulness 0.807 on best balanced run 104356 still below the 0.90 target; qwen2.5:7b can paraphrase unsupported details; single PDF / 81 chunks does not prove multi-doc generalization. Score-based citation fallback in citations.py can surface chunks not tagged in the answer if the model omits [Source N] tags. Policy staleness is not detected automatically.'
    ),
    ('What would you put on a Claude system card for this?', 'Human escalation path?'): para(
        'When balanced mode cannot ground an answer or guard rejects with high confidence, the UI should route to HR—not fabricate policy. Strict mode (faith 1.00, relevancy 0.42) is the audit setting where abstention is preferred. I document both modes in prompts.py via resolve_grounding_mode; sidebar toggle lets users pick. Production gap: no formal ticketing integration yet.'
    ),
    ('Why over-retrieve then rerank instead of top-k=6 directly?', 'Latency cost?'): para(
        'Measured on 5 golden cases (logs/latency_benchmark.json): Chroma p50 49ms, but bge-reranker-large on 30 candidates costs p50 28.5s on CPU—dominates e2e. Full path p50 53.5s / p95 58.8s (rewrite 1.1s, generation 20.2s, guard 859ms). Acceptable for internal HR, not consumer chat. Trade paid off: context precision 0.53→0.80, hit rate ~0.867.'
    ),
    ('Why over-retrieve then rerank instead of top-k=6 directly?', 'Why not hybrid BM25?'): para(
        'Hit rate was already 0.867—recall was not the bottleneck; noisy context was (precision 0.53). Hybrid BM25 is on the roadmap for exact clause IDs and legal vocabulary. Today I use augment_query_with_policy_terms in query_processing.py for deterministic expansion (e.g., resignation→at-will). Measured choice, not theoretical.'
    ),
    ('Why bge-reranker-large over base?', 'GPU path?'): para(
        'Default is CPU in Docker for portability; sentence-transformers cross-encoder runs fine but ~2× slower than GPU. config.py exposes RERANKER_MODEL_NAME; GPU would mainly cut the 200–600ms rerank slice. Graceful degradation if sentence-transformers is missing—vector-only retrieval still works with install hints in retriever.py.'
    ),
    ('Why bge-reranker-large over base?', 'ColBERT alternative?'): para(
        'ColBERT offers late interaction for better lexical matching but adds indexing complexity and another dependency. For 81 chunks on one handbook, cross-encoder rerank on k=30 was the high-ROI choice—+51% context precision in eval logs. At 10k PDFs I would revisit ColBERT or hybrid BM25 before bolting on more neural rankers.'
    ),
    ('How do you handle vocabulary mismatch?', 'Why not HyDE?'): para(
        'HyDE generates a hypothetical document—risky in legal/policy domains where hallucinated statutes pollute retrieval. I chose rule-based augment_query_with_policy_terms plus LLM rewrite (35 words max, fail-open). Measured: notice_resignation relevancy went 0.0→0.80. Eval-driven, not theory.'
    ),
    ('How do you handle vocabulary mismatch?', 'Embedding fine-tune?'): para(
        'Embedding fine-tune needs labeled query-chunk pairs I did not have at solo scale. Keyword expansion plus reranker handled the main failure modes on the 15-case set. Fine-tuning would be next if vocabulary mismatch cases grew after hybrid BM25 and multi-doc indexing.'
    ),
    ('Explain your chunking strategy.', 'Chunk size ablation?'): para(
        'I used 640 tokens with 64 overlap via section-aware SentenceSplitter in indexing.py—section_path metadata preserves legal structure. I did not run a full grid search; fixed char splits were rejected because they split mid-clause. Next ablation would compare 512/64 vs 640/64 on context precision in evaluation.py.'
    ),
    ('Explain your chunking strategy.', 'Multi-doc?'): para(
        'Today: one handbook PDF → 81 chunks in a single Chroma collection with file_hash for incremental SHA-256 indexing. Multi-doc would add collection sharding or metadata filters per department in config.py. enrich_nodes_with_sections() already attaches section_path and page_number for filtering.'
    ),
    ('What is context precision vs hit rate?', 'Context recall formula?'): para(
        'In evaluation.py compute_retrieval_metrics: context recall = matched keywords from relevant_sections / len(relevant_sections). Hit rate asks whether any relevant chunk appears in top-k (~0.867 stable). Precision asks what fraction of retrieved chunks are relevant—that was 0.53 baseline, 0.80 after rerank pipeline in run 104356.'
    ),
    ('What is context precision vs hit rate?', 'NDCG?'): para(
        'I did not implement NDCG in the harness—golden cases use keyword/section rubrics, not graded relevance judgments. For 15 cases LLM judge plus precision/recall was enough to catch the real bug (noisy context). At scale I would add NDCG@k if human graders supply graded chunk relevance.'
    ),
    ('Strict vs balanced grounding — walk me through the trade-off.', 'Can you hit both ≥0.90 and ≥0.75?'): para(
        'Not yet on the same knob: strict mode hit faith 1.00 but relevancy 0.42; best balanced run 104356 is faith 0.807, relv 0.747. The Pareto frontier is real—I document both in README2. Next step is tighter generation claims in prompts.py, not re-enabling full abstention on guard reject.'
    ),
    ('Strict vs balanced grounding — walk me through the trade-off.', 'RLHF analogy?'): para(
        'RLHF trades helpfulness vs harmlessness; my system trades relevancy vs faithfulness with the same structure. Strict mode is harmless but unhelpful (audit); balanced default keeps guard-rejected answers for usefulness (faith 0.807). I log both so we do not optimize one objective silently—alignment teams would call that responsible iteration.'
    ),
    ('Why a second-pass faithfulness guard?', 'Guard false positives?'): para(
        'Balanced guard uses SUPPORTED/UNSUPPORTED; false positives keep borderline paraphrases that score lower on faithfulness—that is partly why faith is 0.807 not 0.90. I trace pre_guard_answer and guard_modified in evaluation.py to quantify impact. Tightening the verifier without collapsing relevancy back toward 0.40 is the active tuning problem.'
    ),
    ('Why a second-pass faithfulness guard?', 'Smaller verifier model?'): para(
        'Same qwen2.5:7b runs gen and guard today—cost is ~200–800ms extra per query. A smaller dedicated verifier could cut latency if correlation with full judge holds on golden set. I would A/B in evaluation.py before shipping; independence from generator would also reduce judge contamination concern.'
    ),
    ('Why mandatory [Source N] tags?', 'What if model invents source numbers?'): para(
        'extract_cited_source_indices in citations.py only maps indices present in context; invented numbers will not match source_nodes from SourceTrackingQueryEngine. If no valid tags, select_citations_for_answer falls back to score ranking capped at 3 sources with ratio 0.55—logged as lower-trust path. Mandatory tags in prompts.py plus Example J (BAD) few-shot reduced omission rate.'
    ),
    ('Why mandatory [Source N] tags?', 'Inline vs footnote?'): para(
        'Inline [Source N] tags keep grounding auditable in the same text employees read—footnotes hide the contract between generation.py and citations.py. select_citations_for_answer mode=cited_in_answer filters UI to only tagged sources, fixing the sick-leave/Holidays citation bug. Streamlit renders the filtered list beside the answer.'
    ),
    ('How did few-shots affect behavior?', 'How many is optimal?'): para(
        'Eight bloated few-shots increased abstention prior and contributed to the 0.40 relevancy regression; trimmed to nine focused examples in prompts.py FEW_SHOT_BALANCED. Optimal count is empirical—I measure via eval runs, not rules of thumb. Quality and abstention balance matter more than raw count.'
    ),
    ('How did few-shots affect behavior?', 'Dynamic few-shot?'): para(
        'Would select examples by retrieval similarity or question category from a small pool—e.g., sick leave cases get sick leave few-shots. Not implemented; static set recovered relv to 0.747. Risk is dynamic selection importing wrong priors; I would gate behind eval with per-category breakdown in evaluation.py.'
    ),
    ('What is normalize_balanced_answer?', 'Regex vs LLM cleanup?'): para(
        'Regex/strip at synthesis time—strips double-ending where partial answer plus INSUFFICIENT_INFO suffix confused judges (scored 0.5). Cheaper and deterministic than another LLM call; applied in generation.py at synthesis, not only in guard. Part of Phase 4 recovery alongside guard fixes.'
    ),
    ('What is normalize_balanced_answer?', 'Other failure patterns?'): para(
        'Guard replacing answers entirely (091001), citation pipeline showing wrong chunks, Chroma false negative in Streamlit—these are separate failure classes. normalize_balanced_answer targets the good-answer-plus-abstention-suffix pattern specifically. Full trace fields help isolate which pattern hit each golden case.'
    ),
    ('Why qwen2.5:7b and not Claude API?', 'Judge contamination?'): para(
        'Same Ollama model for generation and LLM-as-judge in evaluation.py—known limitation I state upfront. Generator and judge sharing weights can correlate scores optimistically. Improvement path: Claude API as judge only while keeping local gen, or a smaller independent verifier model.'
    ),
    ('Why qwen2.5:7b and not Claude API?', 'Claude as judge only?'): para(
        'That is my top eval upgrade: keep offline qwen2.5:7b for generation per project constraints, call Claude for faithfulness/relevancy judging with retrieved context. Would break correlation, improve rubric adherence, and mirror how I would validate a safety-critical change at Anthropic—measure with a stronger independent evaluator.'
    ),
    ('Draw the end-to-end architecture.', 'Where is bottleneck?'): para(
        'End-to-end ~3–8s on CPU: generation 1–5s dominates, then rerank 200–600ms, guard 200–800ms, rewrite 200–800ms. Chroma at 81 chunks is cheap. For internal HR that is acceptable; chat-scale would need caching frequent queries, GPU rerank, or async Streamlit paths.'
    ),
    ('Draw the end-to-end architecture.', 'Async path?'): para(
        'Not implemented—Streamlit and agent.py are synchronous today. Async would help parallelize rewrite plus retrieval warm-up and batch reranker scoring. evaluation.py harness already shares retriever.py pipeline for consistency; async refactor would start at the UI boundary without forking retrieval logic.'
    ),
    ('Why SourceTrackingQueryEngine wrapper?', 'ContextVar vs request state?'): para(
        'ContextVar in citations.py gives per-turn isolation in the ReAct agent without threading request objects through LlamaIndex. begin_citation_turn() resets each policy_search call; record_generation_sources() captures source_nodes from SourceTrackingQueryEngine. Maps cleanly to AgentTurnResult.citations—critical for multi-step agent runs.'
    ),
    ('Why SourceTrackingQueryEngine wrapper?', 'Multi-tool turns?'): para(
        'Each policy_search invocation starts a fresh citation turn via ContextVar; only sources from that generation pass reach the UI. If the agent called multiple tools per turn, I would nest turn IDs or merge citation lists explicitly—today max_iterations=8 with one primary retrieval tool keeps it simple.'
    ),
    ('Why _PostprocessingRetriever wrapper?', 'Fork LlamaIndex?'): para(
        'Wrapper in retriever.py applies rerank plus RelativeScoreThresholdPostprocessor because VectorIndexRetriever ignores postprocessors by default—framework gap, not product gap. Forking LlamaIndex would increase maintenance; a thin wrapper shared with evaluation.py keeps eval and prod identical.'
    ),
    ('Why _PostprocessingRetriever wrapper?', 'Custom retriever class?'): para(
        '_PostprocessingRetriever is effectively a custom retrieve path composing vector retrieval plus postprocessors. Could promote to first-class Retriever subclass if LlamaIndex API stabilizes. Key invariant: k=30 candidates, rerank top 6, 40% score ratio filter per config.py.'
    ),
    ('How does config-driven design help?', 'Secrets management?'): para(
        'Pydantic BaseSettings in config.py loads .env for OLLAMA_MODEL, Chroma paths, flags like ENABLE_CITATION_PIPELINE_LOGGING—12-factor style. No secrets in code; Docker Compose uses host.docker.internal for Ollama. Production would move to vault-injected env vars per environment.'
    ),
    ('How does config-driven design help?', 'Per-tenant config?'): para(
        'Settings could gain department metadata filters and per-tenant Chroma collections—today single tenant, one handbook. get_retrieval_config_summary() already snapshots config per eval run; extending to tenant_id would prevent indexing/agent/eval drift in multi-HR deployment.'
    ),
    ('Agent vs direct query engine — when each?', 'Max iterations?'): para(
        'agent.py sets max_iterations=8 on ReActAgent (LlamaIndex 0.14 agent.run() API). Enough for greet → policy_search → follow-up without infinite tool loops. Direct query engine path skips agent overhead for single-shot Q&A in the Streamlit sidebar.'
    ),
    ('Agent vs direct query engine — when each?', 'Tool hallucination?'): para(
        'policy_search is the only retrieval tool—it wraps build_query_engine() so the agent cannot invent a parallel retrieval path. Meta questions and greetings handled without tool call. Risk remains the model fabricating tool arguments; mitigation is narrow tool surface and eval cases for off-topic queries.'
    ),
    ('How would you scale this to 10k PDFs?', 'Elasticsearch hybrid?'): para(
        'Would add BM25 for clause IDs and exact legal terms while keeping Chroma for semantic recall—hybrid is on README2 roadmap. Shard collections by department; metadata filters on section_path. Today 81 chunks is prototype scale; reranker batching and query cache become necessary at 10k PDFs.'
    ),
    ('How would you scale this to 10k PDFs?', 'Dedicated embed GPU?'): para(
        'Embedding 10k PDFs dominates indexing time on CPU; a dedicated embed service on GPU would shorten build_index() cycles in indexing.py. Incremental SHA-256 file_hash indexing already avoids full re-embed on unchanged files. Separate embed service decouples indexing queue from query serving.'
    ),
    ("What's your evaluation harness architecture?", 'CI gate?'): para(
        'Not wired yet—honest gap. Would run scripts/evaluate.py on PRs, fail if relevancy < 0.70 or faithfulness drops >0.05 vs last main run. 15 cases × ~16 min is heavy for CI; I would start with a 5-case smoke subset plus nightly full golden run. evaluation_results.json append-only log is CI-ready data.'
    ),
    ("What's your evaluation harness architecture?", 'Human eval overlap?'): para(
        'Ran 5-case human overlap (sick_leave, notice_resignation, remote_work, harassment_rules, outside_employment) against run 104356 outputs—rubric in data/eval/HUMAN_EVAL_RUBRIC.md, scores in human_eval_scores.json. Faithfulness: Pearson r 0.95, MAE 0.05, κ@0.5 threshold 1.0, 80% within 0.1. Relevancy: r 0.824, MAE 0.17, κ@0.5 0.0, 60% within 0.1. Biggest disagreement: remote_work relevancy (human 0.5 vs LLM 0.0)—LLM penalized partial answer; human credited policy-grounded content. Agreement script: scripts/compare_human_judge.py → logs/human_judge_agreement.json.'
    ),
    ('Failure mode: three independent axes?', 'Unified metric?'): para(
        'I deliberately avoid one headline number—faithfulness, relevancy, and citation precision fail independently (sick-leave answer with Holidays sources is high faith, zero citation trust). Unified metric hides regressions like strict mode faith 1.00 / relv 0.42. Dashboard with three trends in evaluation_results.json is the operating model.'
    ),
    ('Failure mode: three independent axes?', 'User harm scenario?'): para(
        'Employee asks sick-leave days; model answers plausibly but cites Holidays policy—employee takes wrong PTO action. Hallucinated benefits are another harm vector measured by faithfulness judge. Over-abstention in strict mode harms differently: correct topic, no answer. I default balanced for usefulness; strict for auditors.'
    ),
    ('How did you productionize this?', 'CI/CD?'): para(
        'Docker Compose plus Streamlit plus 94 pytest tests, but no automated pipeline—tests and evaluate.py are manual today. Would add GitHub Actions: pytest on PR, optional eval smoke, image build to registry. entrypoint.sh already waits for Ollama and supports AUTO_INDEX_ON_START.'
    ),
    ('How did you productionize this?', 'K8s?'): para(
        'Docker is deployment-ready for a single replica; K8s would add for multi-tenant HR scale: separate Ollama inference deployment, Chroma or managed vector DB StatefulSet, Streamlit HPA. Host Ollama pattern does not map cleanly to K8s—I would move inference in-cluster with model versioning.'
    ),
    ('Tell me about the citation trust bug.', 'Score fallback risks?'): para(
        'When model omits [Source N], citations.py falls back to score-ranked chunks capped at 3 with ratio 0.55—better than showing all six reranked chunks, but still not generation-linked truth. That is why mandatory tags in prompts.py matter; fallback is degraded mode, logged in citation pipeline stages.'
    ),
    ('Tell me about the citation trust bug.', 'User-reported vs eval?'): para(
        'Caught in QA: sick-leave answer displayed Holidays/Visitors sources—the eval set later added cases to prevent recurrence. tests/test_citations.py guards select_citations_for_answer. Lesson: citation trust bugs are user-visible before aggregate metrics move; pipeline logging chroma_retrieved → post_rerank → query_engine_output helps.'
    ),
    ("Chroma 'no index' false negative — what happened?", 'Multi-worker Chroma?'): para(
        'SharedSystemClient cached stale settings caused Streamlit to show Chunks: 0 while CLI had 81—telemetry conflict. Multi-worker Streamlit would amplify that; fix was reset_chroma_client_cache() and probe_chroma_index() in indexing.py. Managed vector DB or single-worker Chroma behind a service is safer for production.'
    ),
    ("Chroma 'no index' false negative — what happened?", 'Managed vector DB?'): para(
        'At prototype scale embedded Chroma works; production would consider Pinecone/Weaviate/pgvector for HA and consistent client semantics. tests/test_chroma_telemetry.py plus NoOpProductTelemetry addressed local pain. Migration path: same metadata schema (section_path, page_number, file_hash).'
    ),
    ('What observability do you have?', 'OpenTelemetry?'): para(
        'Today: app.log, citation pipeline stages, eval JSON with pre_guard_answer and judge_notes, @timed build_index, plus src/timing.py stage hooks and scripts/benchmark_latency.py (e2e p50 53.5s on 5 golden cases). Gap: no distributed traces or live-traffic dashboards. OpenTelemetry would export the timing spans already instrumented in query_processing.py, retriever.py, and generation.py.'
    ),
    ('What observability do you have?', 'Datadog?'): para(
        'Same gap—no SaaS APM wired. Would ship logs plus OTel traces to Datadog, alert on relevancy drift if nightly eval regresses, monitor Ollama latency. ENABLE_CITATION_PIPELINE_LOGGING already gives structured stage logs for citation debugging.'
    ),
    ('Fallback and degradation strategies?', 'Circuit breaker on Ollama?'): para(
        'Rewrite and guard fail-open to original query / keep answer today; no circuit breaker if Ollama is down—the UI would error on generation. Would add health check on host.docker.internal:11434, exponential backoff, and queue for burst load. Reranker missing already degrades gracefully in retriever.py.'
    ),
    ('Fallback and degradation strategies?', 'Queue under load?'): para(
        'Not implemented—synchronous ~3–8s queries would pile up under concurrent HR traffic. Would add Redis/RQ job queue for async answers in production, keeping Streamlit on polling. Balanced guard reject keeps answer (not abstain) so partial degradation preserves usefulness.'
    ),
    ('Latency breakdown on CPU?', 'What would you cache?'): para(
        'Cache reranked results for top FAQ queries (PTO, dress code, benefits); cache query rewrite output keyed by normalized question. Chroma embedding cache for repeated queries. Invalidation on incremental index update via file_hash in indexing.py. Biggest win is skipping rerank plus gen on exact hits.'
    ),
    ('Latency breakdown on CPU?', 'Batching strategy?'): para(
        'Batch reranker scoring if multiple queries arrive together; batch embed on indexing queue for 10k PDF scale. Single-query path today in _PostprocessingRetriever. Measured rerank p50 is 28.5s on CPU—GPU batching is the highest-ROI cut before caching.'
    ),
    ('Docker: why Ollama on host?', 'Ollama in-compose variant?'): para(
        'Host Ollama reuses existing GPU/RAM and keeps the app container lightweight—docker-compose.yml points to host.docker.internal:11434. In-compose Ollama simplifies onboarding at cost of image size and GPU passthrough complexity. Volumes persist data/storage/logs/hf_cache either way.'
    ),
    ('Docker: why Ollama on host?', 'Model versioning?'): para(
        'Would pin Ollama model tags in config.py (OLLAMA_MODEL=qwen2.5:7b) and document in README; eval runs snapshot model in get_retrieval_config_summary(). Production needs explicit upgrade playbook—re-run 15 golden cases before promoting a new weights version.'
    ),
    ("What's missing for true production?", 'First priority if hired?'): para(
        'CI eval gate with smoke golden subset plus independent Claude-as-judge for faithfulness—closes the loop between research rigor and shipping. Second: hybrid BM25 for legal exact-match. Both directly attack faith 0.807 vs 0.90 gap without sacrificing 0.747 relevancy.'
    ),
    ("What's missing for true production?", '90-day roadmap?'): para(
        'Days 1–30: CI pytest plus eval smoke, OTel spans, Claude judge pilot. Days 31–60: hybrid BM25, 5-case human eval overlap, faithfulness recovery via tighter prompts. Days 61–90: metadata ACL filters, nightly full golden run, p95 latency measurement in staging.'
    ),
    ('Why ReAct agent vs always-on RAG?', 'When does agent skip retrieval?'): para(
        'Greetings, meta questions, and clarifications skip policy_search—the agent answers from system prompt. Substantive policy questions invoke policy_search which runs full build_query_engine() pipeline (k=30, rerank, guard). Same stack, no duplicate retrieval code in agent.py.'
    ),
    ('Why ReAct agent vs always-on RAG?', 'Max iterations 8?'): para(
        'Yes—configured in agent.py create_agent for LlamaIndex 0.14 ReActAgent. Prevents runaway tool loops while allowing multi-step clarify → search → refine. chat_with_memory returns AgentTurnResult(answer, citations) per turn.'
    ),
    ('How is memory used in retrieval?', 'Summarize old turns?'): para(
        'ChatMemoryBuffer holds 5 turns / 3000 tokens; build_retrieval_query() in memory.py expands current question with history—no summarization layer yet. Long conversations could compress older turns via LLM summary, but risk losing policy constraints. Would eval before shipping summarization.'
    ),
    ('How is memory used in retrieval?', 'PII in memory?'): para(
        'HR questions may include employee-specific details in chat history; memory stays in session RAM via ChatMemoryBuffer—not persisted to Chroma. Production would need retention limits, redaction, and per-user session isolation. Sidebar grounding mode toggle does not affect memory content.'
    ),
    ('ContextVar for citations — why?', 'Thread pool?'): para(
        'ContextVar propagates across async tasks in same logical context; thread pool workers need explicit copy_context if we parallelize retrieval. Today synchronous agent.py avoids that pitfall. FastAPI request scope would use similar per-request ContextVar pattern as citations.py.'
    ),
    ('ContextVar for citations — why?', 'FastAPI request scope?'): para(
        'Same pattern maps directly: begin_citation_turn() at request start, record_generation_sources() after generation, return citations in JSON response. ContextVar beats global state for multi-user serving—learned from agent multi-step turns polluting globals.'
    ),
    ('Guard + normalize as workflow validation?', 'Combine into one pass?'): para(
        'Could merge normalize and guard into one LLM call but loses interpretability—guard_modified and pre_guard_answer traces in evaluation.py isolate regressions. Layered validation mirrors Constitutional AI style: prompts → normalize → guard → citation parse in generation.py pipeline. Separate layers let me fix 091001 without rewriting prompts only.'
    ),
    ('Guard + normalize as workflow validation?', 'Constitutional AI parallel?'): para(
        'Layers encode norms: prompts/few-shots set priors, normalize fixes format violations, guard enforces contextual faithfulness, [Source N] enforces verifiability. Like critique-and-revise, but specialized for RAG policy QA. Trade-off naming (balanced vs strict) matches explicit norm selection.'
    ),
    ('LlamaIndex 0.14 migration — what broke?', 'Vendor lock-in?'): para(
        'ReActAgent.from_tools() removal broke Streamlit until I migrated to agent.run() workflow API—upgrade tax is real. Mitigation: thin adapters in agent.py, pin versions, 94 tests. Raw LangGraph would add flexibility at integration cost; policy_search wrapper contains most lock-in surface.'
    ),
    ('LlamaIndex 0.14 migration — what broke?', 'Raw LangGraph?'): para(
        'LangGraph gives explicit state machine for tool loops—attractive for max_iterations and human-in-the-loop. I stayed on LlamaIndex because SourceTrackingQueryEngine, GroundedCompactAndRefine, and retriever wrappers were already integrated. Would prototype LangGraph only if agent complexity grew past single-tool RAG.'
    ),
    ('Faithfulness 0.807 vs 0.90 target — what now?', 'Pareto curve exploration?'): para(
        'I would sweep guard strictness, few-shot count, and TOP_N postprocessor settings, logging faith vs relv per run in evaluation_results.json—already have anchor points: strict (1.00, 0.42), balanced 104356 (0.807, 0.747), regression 091001 (0.94, 0.40). Goal is pushing northeast without hiding either metric.'
    ),
    ('Faithfulness 0.807 vs 0.90 target — what now?', 'Human review queue?'): para(
        'Route guard_modified cases and low faith scores to HR review before showing to employees—production feature not built. Golden set human overlap would calibrate queue thresholds. Balanced mode keeps guard-rejected answers today for relevancy; queue is how I would recover faith toward 0.90 without full abstention.'
    ),
    ('Why not HyDE for query expansion?', 'HyDE with guard?'): para(
        'Hypothetical doc could pass faithfulness guard on its own text while retrieving wrong real chunks—does not fix legal hallucination risk at retrieval stage. My fail-open rewrite plus augment_query_with_policy_terms targets mismatch without inventing statutes. Eval chose deterministic expansion.'
    ),
    ('Why not HyDE for query expansion?', 'Multi-query?'): para(
        'Could issue multiple rewritten queries and fuse results—helps recall at cost of 3× retrieval plus rerank latency. At hit rate 0.867 already, I would prioritize precision filters (rerank top 6, 40% ratio) first. Multi-query is a measured experiment for multi-doc scale, not the current prototype.'
    ),
    ('Why LLM-as-judge vs human labels?', 'Inter-rater agreement?'): para(
        'Measured on 5-case overlap (author rater vs LLM judge, run 104356): faithfulness κ@0.5 = 1.0 (Pearson r 0.95, MAE 0.05); relevancy κ@0.5 = 0.0 (r 0.824, MAE 0.17)—κ collapses when both raters cluster above 0.5 but rank differently. remote_work relevancy drove the gap (human 0.5, LLM 0.0). Faithfulness agreement is strong enough to trust automated gates; relevancy needs human spot-check on borderline partial answers before claiming 0.747 generalizes. Full stats in logs/human_judge_agreement.json.'
    ),
    ('Why LLM-as-judge vs human labels?', 'Claude as judge?'): para(
        'Top methodology upgrade: independent model family judging faithfulness/relevancy with retrieved context, while qwen2.5:7b stays local generator. Reduces contamination from shared weights. Anthropic-aligned validation pattern—stronger judge than policy generator.'
    ),
    ('If you had 2 more weeks?', 'What would you cut?'): para(
        'I would cut Streamlit polish and new agent features before cutting eval or tests. Hybrid BM25 and CI smoke eval deliver more safety ROI than UI extras. 94 pytest tests and 13-run log stay non-negotiable—they are how I caught 091001 and the citation bug.'
    ),
    ('If you had 2 more weeks?', 'Ship vs perfect?'): para(
        'Ship at 0.747 relevancy (within 0.003 of 0.75 target) with documented faith 0.807 gap—honest beats rounded claims. Perfection on faith 1.00 already proved unusable (relv 0.42 strict). Eval trend line is the ship gate, not gut feel.'
    ),
    ('Biggest suboptimal decision?', 'What constraint forced it?'): para(
        'Early focus on maximizing faithfulness without tracking relevancy cost—strict mode looked like victory at faith 1.00 until per-case scores showed 0.42 relevancy. Constraint was treating safety as single metric. Solo bandwidth meant I optimized what I measured first.'
    ),
    ('Biggest suboptimal decision?', 'How communicated to stakeholders?'): para(
        'Documented in README2 and evaluation_results.json with run IDs (073816 strict, 091001 regression, 104356 recovery)—no hiding the Pareto frontier. Interview framing: reduced unsupported claims on golden set, not solved hallucinations. Stakeholders get three metrics, not one green checkbox.'
    ),
    ('How do you approach debugging ML systems?', 'Example trace walkthrough?'): para(
        '091001: aggregate relv 0.40, ctx prec 0.82 → retrieval OK. Open case sick_leave: guard_modified=True, pre_guard_answer had substantive text, post-guard abstention. Trace pointed to apply_faithfulness_guard in generation.py. Single-case repro, fix balanced reject behavior, full 15-case confirm → 0.747.'
    ),
    ('How do you approach debugging ML systems?', 'When to rollback?'): para(
        'Rollback if full golden run drops relevancy >0.10 or faithfulness >0.05 vs last logged main run in evaluation_results.json—or any guard_modified spike across cases. I rolled back prompt/few-shot experiments via append-only logs; formal CI gate would automate that today.'
    ),
    ('How does this project shape your AI philosophy?', 'Disagree with scaling laws?'): para(
        'Scaling helps but this project proved workflow engineering beat model swapping for reliability—qwen2.5:7b with rerank, guard, and citations outperformed naive bigger-context approaches on citation trust. I want scale and alignment tooling together: eval harnesses that survive model upgrades.'
    ),
    ('How does this project shape your AI philosophy?', 'Long-term AGI view?'): para(
        'Systems must expose trade-offs (helpful vs harmless analog) and verifiable outputs ([Source N], eval traces)—not optimize one reward secretly. Long-term capability needs scalable oversight patterns I practiced at 15-case scale. Policy RAG is a microcosm of alignment engineering, not a chatbot side project.'
    ),
    ('What user trust means in this system?', 'Adversarial users?'): para(
        'Prompt injection via handbook PDF is out of scope today; user chat could try to elicit off-policy answers. Faithfulness guard and abstention modes mitigate unsupported claims; strict mode for audits. Would add input filtering and retrieval-only-from-index constraints for adversarial eval cases.'
    ),
    ('What user trust means in this system?', 'Policy staleness?'): para(
        'Incremental indexing via file_hash in indexing.py re-embeds changed PDFs only—no automatic staleness alert if HR updates handbook without re-upload. Production needs version metadata in answers (effective date) and index freshness checks in probe_chroma_index(). Employees trust timestamps as much as citations.'
    ),
    ('Why should Anthropic hire you based on this project?', 'What do you want to learn at Anthropic?'): para(
        'Scalable oversight, eval methodology at frontier-model scale, and how teams ship alignment constraints without collapsing helpfulness—the faith/relevancy frontier I measured locally. I want to work where honest metric reporting and safety documentation are cultural defaults, not interview surprises.'
    ),
    ('Why should Anthropic hire you based on this project?', 'Research vs applied?'): para(
        'This repo is applied with research discipline: 13 eval runs, Pareto analysis, system-card thinking. I enjoy building production workflows (generation.py, citations.py) grounded in measurement (evaluation.py). Anthropic blend of both is the fit—ship grounded systems, publish honest limits.'
    ),
}
