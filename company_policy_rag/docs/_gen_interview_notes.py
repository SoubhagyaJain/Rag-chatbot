#!/usr/bin/env python3
"""Generate docs/interview-notes.html — run once, can delete after."""
from collections import Counter
from html import escape as html_escape
from pathlib import Path

from followup_answers import ANSWERS

OUT = Path(__file__).parent / "interview-notes.html"
OUT_GEN = Path(__file__).parent / "gen-interview-notes.html"

CAT_LABELS = {
    "opening": "Opening & Behavioral",
    "retrieval": "Data & Retrieval",
    "prompts": "Prompts & Grounding",
    "architecture": "Architecture & System Design",
    "production": "Production & MLOps",
    "agentic": "Agentic / Workflow",
    "tradeoffs": "Trade-offs",
    "behavioral": "Behavioral & Impact",
}

def q(title, why, answer, refs, followups, cat="", qnum=0):
    fu_parts = []
    for fi, f in enumerate(followups, 1):
        fid = f"fu-{qnum:02d}-{fi}"
        if (title, f) in ANSWERS:
            fu_parts.append(
                f'<li><a class="fu-link" href="#{fid}">{html_escape(f)}</a></li>'
            )
        else:
            fu_parts.append(f"<li>{html_escape(f)}</li>")
    fu = "".join(fu_parts)
    ans = "".join(f"<li>{html_escape(a)}</li>" for a in answer)
    return f"""
<details class="q-card" data-cat="{cat}" id="q-{qnum:02d}">
  <summary>{html_escape(title)}</summary>
  <div class="q-body">
    <p class="q-why"><strong>Why they ask:</strong> {html_escape(why)}</p>
    <p><strong>Answer outline:</strong></p>
    <ul class="q-answer">{ans}</ul>
    <p class="q-refs"><strong>Reference:</strong> <code>{html_escape(refs)}</code></p>
    <p><strong>Follow-ups:</strong> <span class="fu-hint">(click → scroll to mature answer)</span></p>
    <ul class="q-follow">{fu}</ul>
  </div>
</details>"""

QUESTIONS = []
RAW_QUESTIONS = []
_QNUM = 0

def add_qs(cat, items):
    global _QNUM
    for item in items:
        _QNUM += 1
        RAW_QUESTIONS.append((cat, item[0], item[1], item[2], item[3], item[4]))
        QUESTIONS.append(q(item[0], item[1], item[2], item[3], item[4], cat=cat, qnum=_QNUM))

add_qs("opening", [
    ("Walk me through this project in 60 seconds.", "Tests communication clarity and whether you lead with impact or buzzwords.",
     ["Employees need handbook answers they can trust — not confident hallucinations.",
      "I built an eval-driven RAG: dual-corpus index (308 chunks) → hybrid BM25+dense → rerank → grounded generation → faithfulness guard → code validation → citation-filtered UI.",
      "As solo AI/ML engineer I owned pipeline, eval harness, UI, Docker, and 180 tests.",
      "Policy: relevancy 0.40→0.747 (+87%) on 15-case set (run 104356). Guidebook: full rel 0.700 (run 164848, gate passed); enumeration rel 0.84—measured faithfulness/helpfulness trade-off explicitly."],
     "README2.md, src/evaluation.py", ["What would you do differently?", "Biggest failure?"], "opening"),
    ("Why is this relevant to Anthropic's work?", "Checks alignment with helpful, harmless, honest systems — not just demo RAG.",
     ["Anthropic cares about trade-offs between helpfulness and harmlessness; I measured the analog: relevancy vs faithfulness.",
      "Strict mode hit faithfulness 1.00 but relevancy 0.42 — over-abstention. I named and logged that Pareto frontier.",
      "Citation trust: UI must show only sources that grounded the answer — parallel retrieval destroys verifiability.",
      "Golden-set eval with append-only logs — same discipline as shipping model changes with regression tests."],
     "README2 Acts 2-4, logs/evaluation_results.json", ["How do you calibrate judges?", "What about scalable oversight?"], "opening"),
    ("What was your role and scope?", "Solo project scope and ownership depth.",
     ["Sole builder: architecture, implementation, evaluation, Streamlit UI, Docker, documentation.",
      "End-to-end: indexing, retrieval, prompts, guard, citations, agent, tests.",
      "No separate data labelers — golden set rubrics written by me in golden_dataset.json."],
     "project structure README2", ["How long did it take?", "What would a team split look like?"], "opening"),
    ("Tell me about a time a metric regressed.", "STAR + engineering maturity.",
     ["Situation: run 091001 dropped relevancy to 0.40 despite 0.82 context precision.",
      "Task: recover to ≥0.75 without hiding the failure.",
      "Action: root-caused guard replacing answers with abstention; fixed guard behavior, prompts, query augmentation across 4 phases.",
      "Result: 0.747 relevancy; documented trade-off (faith 0.807)."],
     "generation.py apply_faithfulness_guard, README2 Act 3", ["How did you detect it?", "What guardrails prevent recurrence?"], "opening"),
    ("What would you put on a Claude system card for this?", "Tests safety documentation thinking.",
     ["Intended use: internal HR policy + AI guidebook Q&A on indexed PDFs.",
      "Failure modes: unsupported claims, over-abstention, wrong citation sources, vocabulary mismatch, incomplete enumeration lists, code hallucination.",
      "Mitigations: faithfulness guard, mandatory [Source N], balanced vs strict modes, hybrid BM25, corpus scoping, code validation pipeline, eval harness.",
      "Known limits: 7B local model; guidebook faithfulness 0.629 vs 0.90 target; code-query rel 0.525 on full guidebook run; policy faithfulness 0.807 vs 0.90; CPU e2e p50 53.5s."],
     "prompts.py, generation.py, README.md limitations", ["Residual risk?", "Human escalation path?"], "opening"),
])

add_qs("retrieval", [
    ("Why over-retrieve then rerank instead of top-k=6 directly?", "Classic IR architecture trade-off.",
     ["Bi-encoder (Chroma) optimizes recall; cross-encoder (bge-reranker-large) optimizes precision.",
      "k=30 pool → rerank top 6 → drop below 40% of top score.",
      "Context precision improved 0.53→0.80; hit rate stayed ~0.87."],
     "retriever.py, config RETRIEVAL_CANDIDATE_K=30", ["Latency cost?", "Why not hybrid BM25?"], "retrieval"),
    ("Why bge-reranker-large over base?", "Model selection with measurable trade-off.",
     ["Large model +51% context precision vs baseline in eval logs.",
      "Cost: ~2× rerank latency, ~1-3s cold load, CPU default in Docker.",
      "Graceful degradation if sentence-transformers missing."],
     "config.py RERANKER_MODEL_NAME", ["GPU path?", "ColBERT alternative?"], "retrieval"),
    ("How do you handle vocabulary mismatch?", "Domain retrieval challenge.",
     ["Rule-based augment_query_with_policy_terms() before LLM rewrite.",
      "Examples: resignation→at-will, social media→internet policy.",
      "LLM rewrite 35 words max; fail-open returns original query.",
      "Per-case: notice_resignation relevancy 0.0→0.80."],
     "query_processing.py augment_query_with_policy_terms", ["Why not HyDE?", "Embedding fine-tune?"], "retrieval"),
    ("Explain your chunking strategy.", "Indexing decisions for legal text.",
     ["640 tokens, 64 overlap — section-aware SentenceSplitter.",
      "section_path + page_number + content_type metadata on every chunk.",
      "Incremental SHA-256 file_hash indexing into Chroma.",
      "308 chunks: 80 policy handbook + 228 AI Agents guidebook (0% unknown section_path post-reindex)."],
     "indexing.py get_node_parser, enrich_nodes_with_sections", ["Chunk size ablation?", "Multi-doc?"], "retrieval"),
    ("What is context precision vs hit rate?", "Metric literacy.",
     ["Hit rate: any relevant chunk in top-k (0.87 stable on policy; 1.00 on enum subset 160052).",
      "Context precision: relevant_retrieved / total_retrieved — measures noise in LLM context.",
      "Low precision was the bottleneck on policy; enumeration completeness was guidebook bottleneck."],
     "evaluation.py compute_retrieval_metrics", ["Context recall formula?", "NDCG?"], "retrieval"),
    ("How do you handle enumeration and comprehensive-list questions?", "List completeness is a distinct failure mode from single-fact QA.",
     ["is_comprehensive_list_query() + augment_query_with_guidebook_terms() for named subqueries.",
      "Multi-query retrieval, section-diverse rerank, top_n=12 for comprehensive path.",
      "Enumeration few-shots (Examples M–O) + rules 16b/16c in prompts.py.",
      "Run 164848: full guidebook rel 0.700 (gate passed); enumeration bucket rel 0.84, hit 1.00."],
     "query_processing.py, retriever.py, prompts.py", ["Why multi-query retrieval?", "How do you measure list completeness?"], "retrieval"),
    ("What is cross-corpus bleed and how did you fix it?", "Multi-corpus RAG pitfall — tests retrieval scoping design.",
     ["Shared Chroma collection indexed policy + guidebook; dense retrieval returned handbook chunks for guidebook questions.",
      "Symptom: role_playing_block, planning_block retrieved HR policy sections.",
      "Fix: retrieval_scope.py post-filters by source_file from eval case corpus metadata.",
      "Eval scoped with --corpus guidebook; handbook bleed eliminated on scoped cases."],
     "retrieval_scope.py, golden_dataset_guidebook.json", ["When would you disable corpus scoping?", "Eval signal for bleed?"], "retrieval"),
])

add_qs("prompts", [
    ("Strict vs balanced grounding — walk me through the trade-off.", "Core Anthropic-style alignment question.",
     ["Strict: faithfulness ~1.00, relevancy ~0.42 — audit/compliance use.",
      "Balanced default: faith ~0.81, relv ~0.75 target — internal HR Q&A.",
      "Balanced guard keeps answer on reject; strict replaces with abstention.",
      "I document this explicitly — optimizing one metric hid the regression."],
     "prompts.py resolve_grounding_mode, generation.py _resolve_guard_mode", ["Can you hit both ≥0.90 and ≥0.75?", "RLHF analogy?"], "prompts"),
    ("Why a second-pass faithfulness guard?", "Validation layer design.",
     ["Prompts alone insufficient — model still paraphrases or adds unsupported claims.",
      "Second LLM call: SUPPORTED/UNSUPPORTED (balanced) or YES/NO (strict).",
      "Trace fields: pre_guard_answer, guard_modified for eval.",
      "Context truncated 6000 chars, answer 2000."],
     "generation.py apply_faithfulness_guard", ["Guard false positives?", "Smaller verifier model?"], "prompts"),
    ("Why mandatory [Source N] tags?", "Grounding + UI contract.",
     ["Soft citation rules → model omits tags → score fallback shows wrong sources.",
      "CITATION RULES (MANDATORY) in balanced prompt + agent system prompt.",
      "Example J (BAD) few-shot: correct answer without tags labeled NEVER DO THIS.",
      "Enables select_citations_for_answer() mode=cited_in_answer."],
     "prompts.py, citations.py extract_cited_source_indices", ["What if model invents source numbers?", "Inline vs footnote?"], "prompts"),
    ("How did few-shots affect behavior?", "Prompt prior / abstention analysis.",
     ["8 bloated few-shots increased abstention prior — contributed to 0.40 regression.",
      "Trimmed to 9 focused examples: sick leave, at-will, benefits mapping.",
      "Few-shots set priors as much as rules — measured via eval."],
     "prompts.py FEW_SHOT_BALANCED", ["How many is optimal?", "Dynamic few-shot?"], "prompts"),
    ("What is normalize_balanced_answer?", "Post-generation workflow fix.",
     ["Strips double-ending: partial answer + INSUFFICIENT_INFO suffix.",
      "Judges scored 0.5 on these — silent relevancy killer.",
      "Applied at synthesis time, not only in guard.",
      "Part of Phase 4 recovery to 0.747."],
     "generation.py normalize_balanced_answer", ["Regex vs LLM cleanup?", "Other failure patterns?"], "prompts"),
    ("Why qwen2.5:7b and not Claude API?", "Constraint + honesty.",
     ["Local/offline requirement; Ollama on host, Docker for UI.",
      "Adequate for policy Q&A; complex multi-doc reasoning would need larger model.",
      "Same model for gen + eval judge — note independence limitation."],
     "config.py OLLAMA_MODEL", ["Judge contamination?", "Claude as judge only?"], "prompts"),
    ("Walk me through post-generation code validation.", "Code-heavy RAG needs verifiable outputs, not just prose grounding.",
     ["After faithfulness guard: code_validation.py checks generated code lines against retrieved context.",
      "Heuristic + LLM judge; self-correct once on failure; strip_code fail mode on persistent mismatch.",
      "GenerationTrace logs validation_passed, fallback_reason for eval.",
      "Baseline 132316: 0% pass, 14% fallback. Tuned 143246: pass 100%, fallback 0%."],
     "src/code_validation.py, generation.py GenerationTrace", ["False positive vs false negative?", "Self-correct once — why?"], "prompts"),
])

add_qs("architecture", [
    ("Draw the end-to-end architecture.", "System design whiteboard.",
     ["PDF → section chunking → Chroma → query rewrite/augment → k=30 → rerank → score filter → GroundedCompactAndRefine → normalize → guard → SourceTracking → citation filter → Streamlit.",
      "Agent path: ReAct calls policy_search tool — same pipeline.",
      "Single config.py source of truth."],
     "README2 architecture block", ["Where is bottleneck?", "Async path?"], "architecture"),
    ("Why SourceTrackingQueryEngine wrapper?", "Proxy pattern for observability.",
     ["Records source_nodes from generation — not a second retrieval.",
      "Feeds citations.py via ContextVar per agent turn.",
      "Fixes sick-leave answer showing Holidays/Visitors sources."],
     "generation.py SourceTrackingQueryEngine, citations.py", ["ContextVar vs request state?", "Multi-tool turns?"], "architecture"),
    ("Why _PostprocessingRetriever wrapper?", "LlamaIndex framework gap.",
     ["VectorIndexRetriever ignores postprocessors by default.",
      "Wrapper applies rerank + RelativeScoreThresholdPostprocessor on retrieve path.",
      "Shared with eval harness for consistency."],
     "retriever.py _PostprocessingRetriever", ["Fork LlamaIndex?", "Custom retriever class?"], "architecture"),
    ("How does config-driven design help?", "Production maintainability.",
     ["Pydantic BaseSettings — .env overrides, no magic numbers in code.",
      "get_retrieval_config_summary() snapshotted per eval run.",
      "Prevents indexing/agent/eval drift."],
     "config.py Settings", ["Secrets management?", "Per-tenant config?"], "architecture"),
    ("Agent vs direct query engine — when each?", "Orchestration scope.",
     ["Agent: multi-turn chat, meta questions, tool decision.",
      "policy_search wraps full build_query_engine() — no forked retrieval.",
      "LlamaIndex 0.14: ReActAgent + agent.run(), not deprecated from_tools.",
      "AgentTurnResult(answer, citations)."],
     "agent.py chat_with_memory", ["Max iterations?", "Tool hallucination?"], "architecture"),
    ("How would you scale this to 10k PDFs?", "Growth path — honest limits.",
     ["Shard Chroma collections; metadata filters per department.",
      "Async indexing queue; separate embed service.",
      "Reranker batching; cache frequent queries.",
      "Today: single collection, 308 chunks (dual corpus) — prototype scale; hybrid BM25 shipped."],
     "indexing.py build_index incremental", ["Elasticsearch hybrid?", "Dedicated embed GPU?"], "architecture"),
    ("What's your evaluation harness architecture?", "Eval systems design — Anthropic cares deeply.",
     ["60 cases JSON (15 policy + 35 guidebook + subsets) → retrieve → generate_grounded_answer_with_trace → LLM judge.",
      "Append-only evaluation_results.json — 13 policy runs + Track A guidebook runs logged.",
      "Config snapshots per run; guard_modified + code_validation trace fields isolate regressions.",
      "~20 min full 60-case run on CPU."],
     "evaluation.py run_evaluation, scripts/evaluate.py", ["CI gate?", "Human eval overlap?"], "architecture"),
    ("Failure mode: three independent axes?", "Safety framing.",
     ["Faithfulness: unsupported claims.",
      "Relevancy: over-abstention or off-topic.",
      "Citation precision: right answer, wrong sources.",
      "Each measured separately — optimizing one hides others."],
     "README2 problem section", ["Unified metric?", "User harm scenario?"], "architecture"),
])

add_qs("production", [
    ("How did you productionize this?", "MLOps maturity for solo project.",
     ["Streamlit UI with index diagnostics; Docker Compose + host Ollama.",
      "entrypoint.sh: Ollama wait 60s, optional AUTO_INDEX_ON_START.",
      "180 pytest tests; probe_chroma_index() for health.",
      "Logs: app.log, citation pipeline, eval JSON, GenerationTrace."],
     "Dockerfile, docker/entrypoint.sh, app/streamlit_app.py", ["CI/CD?", "K8s?"], "production"),
    ("Tell me about the citation trust bug.", "Production battle story.",
     ["UI ran separate retriever.retrieve() — showed all 6 reranked chunks.",
      "User saw sick-leave answer with Holidays/Visitors sources.",
      "Fix: generation-linked citations only; mandatory [Source N] tags.",
      "tests/test_citations.py guards regression."],
     "citations.py select_citations_for_answer", ["Score fallback risks?", "User-reported vs eval?"], "production"),
    ("Chroma 'no index' false negative — what happened?", "Silent failure debugging.",
     ["Streamlit showed Chunks: 0; CLI had 308.",
      "SharedSystemClient cached stale settings; telemetry conflict.",
      "Fix: reset_chroma_client_cache(), probe_chroma_index(), NoOpProductTelemetry.",
      "tests/test_chroma_telemetry.py."],
     "indexing.py probe_chroma_index, chroma_telemetry.py", ["Multi-worker Chroma?", "Managed vector DB?"], "production"),
    ("What observability do you have?", "Ops depth.",
     ["Citation pipeline logging stages: chroma_retrieved → post_rerank → query_engine_output.",
      "Eval trace: pre_guard_answer, judge_notes.",
      "@timed build_index; src/timing.py + benchmark_latency.py (e2e p50 53.5s).",
      "Gap: no live-traffic dashboards, no drift alerts."],
     "citations.py log_retrieval_stage, ENABLE_CITATION_PIPELINE_LOGGING", ["OpenTelemetry?", "Datadog?"], "production"),
    ("Fallback and degradation strategies?", "Reliability engineering.",
     ["Query rewrite fail-open → original query.",
      "Reranker missing → vector-only + install hints.",
      "Citation no tags → score fallback capped at 3 sources, ratio 0.55.",
      "Balanced guard reject → keep answer (not abstain)."],
     "query_processing.py, retriever.py, citations.py", ["Circuit breaker on Ollama?", "Queue under load?"], "production"),
    ("Latency breakdown on CPU?", "Performance honesty.",
     ["Measured 5 golden cases: rewrite p50 1.1s, Chroma 49ms, rerank 28.5s, gen 20.2s, guard 859ms.",
      "E2E p50 53.5s / p95 58.8s — acceptable for internal HR, not chat-scale.",
      "Reranker dominates (~58% e2e); cold load ~10s on first query after process start."],
     "README.md latency section", ["What would you cache?", "Batching strategy?"], "production"),
    ("Docker: why Ollama on host?", "Deployment trade-off.",
     ["Reuse existing GPU/RAM setup; keep app container lightweight.",
      "host.docker.internal:11434; volumes for data/storage/logs/hf_cache.",
      "Streamlit binds 0.0.0.0 — required for port mapping."],
     "docker-compose.yml, .streamlit/config.toml", ["Ollama in-compose variant?", "Model versioning?"], "production"),
    ("What's missing for true production?", "Calibration — Anthropic values honesty.",
     ["No CI/CD pipeline; no automated eval gate on PR — Phase 4 CI is next (guidebook gate passed at 0.700 on run 164848).",
      "No production drift monitoring; no ACL per user.",
      "Benchmarked p50/p95 on golden set (53.5s/58.8s e2e) — no live-traffic metrics yet; CPU-only default.",
      "Hybrid BM25 shipped; guidebook rel gate passed; faithfulness 0.629 still below 0.90 target."],
     "README2 open items", ["First priority if hired?", "90-day roadmap?"], "production"),
])

add_qs("agentic", [
    ("Why ReAct agent vs always-on RAG?", "Workflow orchestration.",
     ["Agent decides when to call policy_search vs handle greetings.",
      "Multi-turn: ChatMemoryBuffer 5 turns, 3000 tokens.",
      "Same retrieval stack inside tool — no duplication."],
     "agent.py build_policy_search_tool", ["When does agent skip retrieval?", "Max iterations 8?"], "agentic"),
    ("How is memory used in retrieval?", "Context assembly.",
     ["build_retrieval_query() expands current question with history.",
      "Memory does not alter Chroma index — only query formulation.",
      "Sidebar toggle grounding mode per session."],
     "memory.py build_retrieval_query", ["Summarize old turns?", "PII in memory?"], "agentic"),
    ("ContextVar for citations — why?", "Async-safe per-turn state.",
     ["begin_citation_turn() resets; record_generation_sources() on each policy_search.",
      "Survives agent multi-step without global pollution.",
      "Maps to AgentTurnResult.citations."],
     "citations.py ContextVar", ["Thread pool?", "FastAPI request scope?"], "agentic"),
    ("Guard + normalize as workflow validation?", "Multi-layer verification.",
     ["Layer 1: prompts + few-shots.",
      "Layer 2: normalize_balanced_answer at synthesis.",
      "Layer 3: faithfulness guard second LLM pass.",
      "Layer 4: citation tag parsing — UI trust."],
     "generation.py pipeline", ["Combine into one pass?", "Constitutional AI parallel?"], "agentic"),
    ("LlamaIndex 0.14 migration — what broke?", "Framework upgrade battle.",
     ["ReActAgent.from_tools() removed → agent.run() workflow API.",
      "Streamlit failed with cryptic from_tools error.",
      "Pin versions or test UI after upgrades."],
     "agent.py create_agent", ["Vendor lock-in?", "Raw LangGraph?"], "agentic"),
])

add_qs("tradeoffs", [
    ("Faithfulness 0.807 vs 0.90 target — what now?", "Honest gap + plan.",
     ["Trade-off from keeping guard-rejected answers for relevancy.",
      "Not re-enabling full abstention on reject.",
      "Next: tighter generation claims, not stricter guard replacement.",
      "Say 0.747 relevancy (within 0.003 of 0.75) — don't round up."],
     "README2 open items, run 104356", ["Pareto curve exploration?", "Human review queue?"], "tradeoffs"),
    ("Why not HyDE for query expansion?", "Alternative considered.",
     ["HyDE generates hypothetical doc — risky in legal/policy (hallucinated statutes).",
      "Chose keyword rewrite + deterministic term expansion.",
      "Measured via eval, not theory."],
     "query_processing.py docstring", ["HyDE with guard?", "Multi-query?"], "tradeoffs"),
    ("Why LLM-as-judge vs human labels?", "Eval methodology trade-off.",
     ["Solo project — 15 cases, iterate fast.",
      "Judge uses retrieved context for relevancy — reduces false penalties.",
      "Domain rubric: at-will resignation, correct abstention scored fairly.",
      "Limitation: same model family — note in interview."],
     "evaluation.py judge_answer_relevancy", ["Inter-rater agreement?", "Claude as judge?"], "tradeoffs"),
    ("If you had 2 more weeks?", "Prioritization.",
     ["Phase 4 CI: pytest + stratified eval smoke (enumeration + code cases).",
      "Full guidebook re-eval to confirm rel ≥ 0.70 before locking CI thresholds.",
      "Independent Claude-as-judge; faithfulness recovery without relevancy loss.",
      "Live-traffic p95 dashboard (golden-set benchmark done: 58.8s e2e p95)."],
     "README2 roadmap", ["What would you cut?", "Ship vs perfect?"], "tradeoffs"),
    ("Biggest suboptimal decision?", "Credibility question.",
     ["Initially maximized faithfulness without measuring relevancy cost.",
      "Strict mode looked great (1.00) until we read per-case scores.",
      "Lesson: always log multi-metric dashboards before declaring victory."],
     "run 073816", ["What constraint forced it?", "How communicated to stakeholders?"], "tradeoffs"),
])

add_qs("behavioral", [
    ("How do you approach debugging ML systems?", "Process question.",
     ["Measure first: evaluation_results.json trend log caught 091001.",
      "Isolate stage: retrieval metrics vs guard_modified flag.",
      "Reproduce with single golden case before full 15-case run.",
      "Add pytest guard for each fixed regression."],
     "tests/, logs/evaluation_results.json", ["Example trace walkthrough?", "When to rollback?"], "behavioral"),
    ("How does this project shape your AI philosophy?", "Culture fit for Anthropic.",
     ["Workflow engineering beats model swapping for reliability.",
      "Name trade-offs explicitly — faithfulness vs helpfulness is real.",
      "Citation trust is a first-class metric, not an afterthought.",
      "Eval harness is the project's memory — ship nothing without trend data."],
     "README2 lessons learned", ["Disagree with scaling laws?", "Long-term AGI view?"], "behavioral"),
    ("What user trust means in this system?", "Harm / UX framing.",
     ["Wrong sources worse than fewer sources — destroys verifyability.",
      "Employees act on sick-leave policy — hallucinated days = real harm.",
      "Balanced mode default for usefulness; strict for auditors."],
     "citation trust fix README2 Act 6", ["Adversarial users?", "Policy staleness?"], "behavioral"),
    ("Why should Anthropic hire you based on this project?", "Closing pitch.",
     ["Demonstrated eval discipline and honest metric reporting on solo build.",
      "Built multi-layer grounding workflow — not a notebook demo.",
      "Documented failures (0.40 regression, citation bug) with fixes and tests.",
      "Think in alignment terms: trade-offs, verifiability, abstention calibration."],
     "Full repo SoubhagyaJain/Rag-chatbot", ["What do you want to learn at Anthropic?", "Research vs applied?"], "behavioral"),
])

def qb_for(cat):
    return "".join(x for x in QUESTIONS if f'data-cat="{cat}"' in x)

QB_OPENING = qb_for("opening")
QB_RETRIEVAL = qb_for("retrieval")
QB_PROMPTS = qb_for("prompts")
QB_ARCHITECTURE = qb_for("architecture")
QB_PRODUCTION = qb_for("production")
QB_AGENTIC = qb_for("agentic")
QB_TRADEOFFS = qb_for("tradeoffs")
QB_BEHAVIORAL = qb_for("behavioral")

def build_followup_bank():
    blocks = []
    for qnum, row in enumerate(RAW_QUESTIONS, 1):
        cat, title, *_rest, followups = row
        blocks.append(
            f'<article class="fu-group" id="fu-group-{qnum:02d}">'
            f'<h3 class="fu-parent"><span class="fu-parent-num">Q{qnum:02d}</span> '
            f'{html_escape(title)}</h3>'
            f'<p class="fu-parent-cat">{CAT_LABELS[cat]} · '
            f'<a href="#q-{qnum:02d}">↑ parent question</a></p>'
        )
        for fi, f in enumerate(followups, 1):
            fid = f"fu-{qnum:02d}-{fi}"
            ans = ANSWERS.get((title, f), "<p><em>Answer pending.</em></p>")
            blocks.append(
                f'<div class="fu-answer-block" id="{fid}">'
                f'<h4 class="fu-q">{html_escape(f)}</h4>'
                f'<div class="fu-answer">{ans}</div>'
                f'<a class="fu-back" href="#q-{qnum:02d}">↑ Back to Q{qnum:02d}</a>'
                f"</div>"
            )
        blocks.append("</article>")
    return "\n".join(blocks)

FOLLOWUP_BANK = build_followup_bank()

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def source_card(idx, cat, title, why, answer, refs, followups):
    ans = "".join(f"<li>{esc(a)}</li>" for a in answer)
    fu = "".join(f"<li>{esc(f)}</li>" for f in followups)
    return f"""
<details class="source-card" data-cat="{cat}">
  <summary><span class="src-num">Q{idx:02d}</span> {esc(title)} <span class="src-cat">{CAT_LABELS[cat]}</span></summary>
  <div class="src-body">
    <div class="src-field"><span class="src-label">Category key</span><code>{cat}</code></div>
    <div class="src-field"><span class="src-label">Why they ask</span><p>{esc(why)}</p></div>
    <div class="src-field"><span class="src-label">Answer outline</span><ul>{ans}</ul></div>
    <div class="src-field"><span class="src-label">Code reference</span><code>{esc(refs)}</code></div>
    <div class="src-field"><span class="src-label">Follow-ups</span><ul class="src-follow">{fu}</ul></div>
  </div>
</details>"""

SOURCE_BANK = "\n".join(
    source_card(i, cat, title, why, answer, refs, followups)
    for i, (cat, title, why, answer, refs, followups) in enumerate(RAW_QUESTIONS, 1)
)

def cat_stats_rows():
    counts = Counter(cat for cat, *_ in RAW_QUESTIONS)
    rows = []
    for cat in CAT_LABELS:
        rows.append(
            f'<tr><td>{CAT_LABELS[cat]}</td><td><code>{cat}</code></td>'
            f'<td class="num">{counts[cat]}</td></tr>'
        )
    return "\n".join(rows)

CAT_STATS_ROWS = cat_stats_rows()

CODE_SNIPPET = esc("""def q(title, why, answer, refs, followups, cat=""):
    # Builds <details class="q-card"> accordion HTML per question

def add_qs(cat, items):
    for item in items:
        QUESTIONS.append(q(item[0], item[1], item[2], item[3], item[4], cat=cat))

def qb_for(cat):
    return "".join(x for x in QUESTIONS if f'data-cat="{cat}"' in x)

# Inject into f-string template — never break the f-string with ''' + ... + '''
html = f'''...{{QB_OPENING}}...{{len(QUESTIONS)}} questions...'''""")

# Read base CSS from project-plans - we'll inline a condensed version
CSS_EXTRA = """
    .anthropic-badge { background: linear-gradient(135deg, #d4a57433, #c9956a22); border: 1px solid #c9956a55; color: #92400e; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 0.3rem 0.65rem; border-radius: 999px; }
    .pitch-box { background: rgba(255,255,255,0.55); border-left: 4px solid var(--accent); padding: 1.25rem 1.5rem; border-radius: var(--radius-sm); margin: 1rem 0; font-size: 1.05rem; line-height: 1.7; color: var(--text-primary); }
    .pitch-box em { color: var(--text-secondary); font-style: normal; font-size: 0.9rem; display: block; margin-top: 0.75rem; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
    .btn { padding: 0.5rem 1rem; border-radius: 10px; border: 1px solid var(--glass-border); background: rgba(255,255,255,0.6); cursor: pointer; font-size: 0.85rem; font-weight: 600; color: var(--text-primary); min-height: 44px; }
    .btn:hover { background: var(--accent-light); }
    .q-bank-controls { margin-bottom: 1rem; }
    .q-card { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--radius-sm); margin-bottom: 0.6rem; overflow: hidden; }
    .q-card summary { padding: 1rem 1.25rem; cursor: pointer; font-weight: 600; color: var(--text-primary); list-style: none; min-height: 44px; display: flex; align-items: center; }
    .q-card summary::-webkit-details-marker { display: none; }
    .q-card summary::before { content: '+'; margin-right: 0.75rem; color: var(--accent); font-weight: 700; }
    .q-card[open] summary::before { content: '−'; }
    .q-body { padding: 0 1.25rem 1.25rem; border-top: 1px solid var(--glass-border-subtle); }
    .q-why { font-size: 0.92rem; color: var(--text-muted); margin: 0.75rem 0; }
    .q-answer, .q-follow { margin: 0.5rem 0 0.75rem 1.25rem; color: var(--text-secondary); font-size: 0.92rem; }
    .q-refs { font-size: 0.85rem; margin-top: 0.5rem; }
    .cat-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin: 1.5rem 0 0.75rem; }
    ul.bullets { margin: 0.75rem 0 1rem 1.25rem; color: var(--text-secondary); }
    ul.bullets li { margin-bottom: 0.35rem; }
    .trade-table td:first-child { font-weight: 600; color: var(--text-primary); white-space: nowrap; }
    .soundbite { font-style: italic; color: var(--text-primary); border-left: 3px solid var(--metric); padding-left: 1rem; margin: 0.75rem 0; }
    .red-flag { color: #dc2626; font-weight: 600; }
    .comp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .comp-card { padding: 1.25rem; background: rgba(255,255,255,0.35); border-radius: var(--radius-sm); border: 1px solid var(--glass-border-subtle); }
    .comp-card h4 { margin: 0 0 0.5rem; text-transform: none; letter-spacing: 0; font-size: 1rem; }
    .fu-hint { font-size: 0.75rem; color: var(--text-muted); font-weight: 400; }
    .fu-link { color: var(--accent); text-decoration: none; font-weight: 600; border-bottom: 1px dashed rgba(13,148,136,0.5); }
    .fu-link:hover { background: var(--accent-light); border-bottom-style: solid; }
    .fu-group { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--glass-border-subtle); }
    .fu-parent { font-size: 1.08rem; margin: 0 0 0.25rem; color: var(--text-primary); font-family: var(--font-serif); }
    .fu-parent-num { font-size: 0.7rem; font-weight: 700; color: var(--accent); background: var(--accent-light); padding: 0.15rem 0.45rem; border-radius: 6px; margin-right: 0.5rem; font-family: var(--font-sans); }
    .fu-parent-cat { font-size: 0.78rem; color: var(--text-muted); margin: 0 0 1rem; }
    .fu-parent-cat a { color: var(--accent); }
    .fu-answer-block { scroll-margin-top: 5.5rem; background: rgba(255,255,255,0.48); border: 1px solid var(--glass-border); border-radius: var(--radius-sm); padding: 1.25rem 1.35rem; margin: 0.75rem 0; transition: box-shadow 0.25s, border-color 0.25s; }
    .fu-answer-block:target { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); background: rgba(255,255,255,0.72); }
    .fu-q { font-size: 0.98rem; color: var(--text-primary); margin: 0 0 0.75rem; font-weight: 600; }
    .fu-answer p { color: var(--text-secondary); margin-bottom: 0.65rem; line-height: 1.7; font-size: 0.95rem; }
    .fu-answer p:last-child { margin-bottom: 0; }
    .fu-back { display: inline-block; margin-top: 0.75rem; font-size: 0.8rem; color: var(--accent); font-weight: 600; text-decoration: none; }
    .fu-back:hover { text-decoration: underline; }
"""

CSS_GEN = """
    .gen-hero { margin-bottom: 2rem; }
    .cmd-box { background: #0f172a; color: #e2e8f0; padding: 1rem 1.25rem; border-radius: var(--radius-sm); font-family: 'IBM Plex Sans', monospace; font-size: 0.88rem; overflow-x: auto; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1); }
    .cmd-box code { color: #5eead4; background: none; padding: 0; }
    .file-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin: 1rem 0; }
    .file-pill { padding: 0.75rem 1rem; background: rgba(255,255,255,0.45); border: 1px solid var(--glass-border); border-radius: var(--radius-sm); font-size: 0.85rem; }
    .file-pill strong { display: block; color: var(--accent); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
    .file-pill.out { border-color: var(--success); background: var(--success-light); }
    .code-panel { background: #0f172a; color: #cbd5e1; padding: 1.25rem; border-radius: var(--radius-sm); font-family: 'IBM Plex Sans', monospace; font-size: 0.8rem; line-height: 1.55; overflow-x: auto; white-space: pre; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.08); }
    .source-card { background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: var(--radius-sm); margin-bottom: 0.5rem; overflow: hidden; }
    .source-card summary { padding: 0.85rem 1.1rem; cursor: pointer; font-weight: 600; font-size: 0.92rem; list-style: none; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; min-height: 44px; }
    .source-card summary::-webkit-details-marker { display: none; }
    .src-num { font-size: 0.7rem; font-weight: 700; color: var(--accent); background: var(--accent-light); padding: 0.15rem 0.45rem; border-radius: 6px; }
    .src-cat { font-size: 0.68rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-left: auto; }
    .src-body { padding: 0 1.1rem 1.1rem; border-top: 1px solid var(--glass-border-subtle); font-size: 0.88rem; }
    .src-field { margin: 0.75rem 0; }
    .src-label { display: block; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); margin-bottom: 0.35rem; }
    .src-field ul { margin: 0.25rem 0 0 1.1rem; color: var(--text-secondary); }
    .src-follow li { color: var(--text-muted); font-size: 0.85rem; }
    .warn-box { background: rgba(254, 243, 199, 0.55); border-left: 4px solid var(--metric); padding: 1rem 1.25rem; border-radius: var(--radius-sm); margin: 1rem 0; font-size: 0.92rem; color: var(--text-secondary); }
    .warn-box strong { color: #92400e; }
    table.num-table td.num { text-align: center; font-weight: 700; color: var(--accent); }
    .step-row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 1rem 0; }
    .step-chip { padding: 0.4rem 0.85rem; background: rgba(255,255,255,0.5); border: 1px solid var(--glass-border); border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .step-arrow { color: var(--accent); font-weight: 700; }
"""

# Load CSS from project-plans.html
plans = (Path(__file__).parent / "project-plans.html").read_text(encoding="utf-8")
css_start = plans.index("<style>") + 7
css_end = plans.index("</style>")
base_css = plans[css_start:css_end]

SVG_PIPELINE = '''<svg class="b3b-svg" viewBox="0 0 700 900" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#fff" opacity="0.06"/></pattern><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#FFFF00"/></marker></defs><rect width="700" height="900" fill="#0c0e14"/><rect width="700" height="900" fill="url(#dots)"/><text class="phase-label" x="40" y="40" fill="#58C4DD">INGEST</text><rect x="200" y="55" width="300" height="44" rx="8" fill="#58C4DD" fill-opacity="0.12" stroke="#58C4DD"/><text class="node-label" x="350" y="82" text-anchor="middle">PDF → Index 640/64 → Chroma</text><path class="flow-arrow" d="M350 99 L350 125" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar)"/><text class="phase-label" x="40" y="155" fill="#83C167">RETRIEVE</text><rect x="200" y="130" width="300" height="44" rx="8" fill="#83C167" fill-opacity="0.12" stroke="#83C167"/><text class="node-label" x="350" y="157" text-anchor="middle">Rewrite + k=30 → Rerank → Filter</text><path class="flow-arrow" d="M350 174 L350 200" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar)"/><text class="phase-label" x="40" y="230" fill="#FF862F">GENERATE</text><rect x="200" y="205" width="300" height="44" rx="8" fill="#FF862F" fill-opacity="0.12" stroke="#FF862F"/><text class="node-label" x="350" y="232" text-anchor="middle">GroundedCompactAndRefine</text><path class="flow-arrow" d="M350 249 L350 275" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar)"/><rect x="200" y="280" width="300" height="44" rx="8" fill="#FF862F" fill-opacity="0.12" stroke="#FF862F"/><text class="node-label" x="350" y="307" text-anchor="middle">normalize → Faithfulness Guard</text><path class="flow-arrow" d="M350 324 L350 350" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar)"/><text class="phase-label" x="40" y="380" fill="#9A72AC">DELIVER</text><rect x="200" y="355" width="300" height="44" rx="8" fill="#9A72AC" fill-opacity="0.12" stroke="#9A72AC"/><text class="node-label" x="350" y="382" text-anchor="middle">SourceTracking → [Source N] UI</text><path class="flow-arrow" d="M350 399 L350 430" stroke="#58C4DD" stroke-width="2.5" marker-end="url(#ar)"/><rect x="175" y="435" width="350" height="40" rx="20" fill="#58C4DD" fill-opacity="0.2" stroke="#58C4DD" stroke-width="2"/><text class="node-label" x="350" y="460" text-anchor="middle" font-weight="600">Streamlit + Docker</text><text font-family="IBM Plex Sans,sans-serif" font-size="10" fill="rgba(255,255,255,0.4)" x="350" y="520" text-anchor="middle">Solo AI/ML Engineer · config.py single source of truth</text></svg>'''

SVG_TRADEOFF = '''<svg class="b3b-svg" viewBox="0 0 800 360" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="d2" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#fff" opacity="0.06"/></pattern><marker id="ar2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#FFFF00"/></marker></defs><rect width="800" height="360" fill="#0c0e14"/><rect width="800" height="360" fill="url(#d2)"/><circle cx="400" cy="50" r="32" fill="#FFFF00" fill-opacity="0.12" stroke="#FFFF00" stroke-width="2"/><text class="node-label" x="400" y="55" text-anchor="middle">Query</text><path class="flow-arrow" d="M370 78 C280 110, 180 140, 160 165" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar2)"/><path class="flow-arrow" d="M430 78 C520 110, 620 140, 640 165" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar2)"/><rect x="40" y="165" width="320" height="160" rx="12" fill="#FC6255" fill-opacity="0.08" stroke="#FC6255"/><text class="phase-label" x="60" y="195" fill="#FC6255">Strict · Audit</text><text class="node-label" x="60" y="230">Faithfulness 1.00</text><text class="node-label" x="60" y="260">Relevancy 0.42</text><text class="node-sub" x="60" y="290">Over-abstention trap</text><rect x="440" y="165" width="320" height="160" rx="12" fill="#83C167" fill-opacity="0.08" stroke="#83C167"/><text class="phase-label" x="460" y="195" fill="#83C167">Balanced · Default</text><text class="node-label" x="460" y="230">Faithfulness 0.807</text><text class="node-label" x="460" y="260">Relevancy 0.747</text><text class="node-sub" x="460" y="290">Anthropic analog: helpful + honest</text><text font-family="IBM Plex Sans,sans-serif" font-size="10" fill="rgba(255,255,255,0.35)" x="400" y="345" text-anchor="middle">Name the Pareto frontier — do not optimize one metric silently</text></svg>'''

SVG_EVAL = '''<svg class="b3b-svg" viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="d3" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#fff" opacity="0.06"/></pattern><marker id="ar3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#FFFF00"/></marker></defs><rect width="800" height="320" fill="#0c0e14"/><rect width="800" height="320" fill="url(#d3)"/><rect x="30" y="120" width="120" height="50" rx="8" fill="#58C4DD" fill-opacity="0.15" stroke="#58C4DD"/><text class="node-label" x="90" y="150" text-anchor="middle">Change</text><path class="flow-arrow" d="M150 145 L190 145" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar3)"/><rect x="190" y="120" width="140" height="50" rx="8" fill="#83C167" fill-opacity="0.15" stroke="#83C167"/><text class="node-label" x="260" y="142" text-anchor="middle">evaluate.py</text><text class="node-sub" x="260" y="158" text-anchor="middle">60 cases</text><path class="flow-arrow" d="M330 145 L370 145" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar3)"/><rect x="370" y="110" width="150" height="70" rx="8" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F"/><text class="node-label" x="445" y="138" text-anchor="middle">LLM Judge</text><text class="node-sub" x="445" y="158" text-anchor="middle">faith + relv</text><path class="flow-arrow" d="M520 145 L560 145" stroke="#FFFF00" stroke-width="2" marker-end="url(#ar3)"/><rect x="560" y="120" width="200" height="50" rx="8" fill="#9A72AC" fill-opacity="0.15" stroke="#9A72AC"/><text class="node-label" x="660" y="150" text-anchor="middle">evaluation_results.json</text><path d="M660 170 C660 220, 90 220, 90 175" stroke="#58C4DD" stroke-width="1.5" stroke-dasharray="6 4" fill="none" marker-end="url(#ar3)"/><text font-family="IBM Plex Sans,sans-serif" font-size="10" fill="#FC6255" x="400" y="260" text-anchor="middle">Policy 091001: relv 0.40→0.747 · Guidebook 164848: rel 0.700</text></svg>'''

SVG_GEN = '''<svg class="b3b-svg" viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="dg" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#fff" opacity="0.06"/></pattern><marker id="arg" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#FFFF00"/></marker></defs><rect width="900" height="300" fill="#0c0e14"/><rect width="900" height="300" fill="url(#dg)"/><rect x="20" y="110" width="130" height="56" rx="8" fill="#58C4DD" fill-opacity="0.15" stroke="#58C4DD"/><text class="node-label" x="85" y="135" text-anchor="middle">_gen_interview</text><text class="node-sub" x="85" y="152" text-anchor="middle">notes.py</text><path class="flow-arrow" d="M150 138 L185 138" stroke="#FFFF00" stroke-width="2" marker-end="url(#arg)"/><rect x="185" y="110" width="120" height="56" rx="8" fill="#83C167" fill-opacity="0.15" stroke="#83C167"/><text class="node-label" x="245" y="142" text-anchor="middle">add_qs()</text><path class="flow-arrow" d="M305 138 L340 138" stroke="#FFFF00" stroke-width="2" marker-end="url(#arg)"/><rect x="340" y="110" width="100" height="56" rx="8" fill="#FF862F" fill-opacity="0.15" stroke="#FF862F"/><text class="node-label" x="390" y="142" text-anchor="middle">q()</text><path class="flow-arrow" d="M440 138 L475 138" stroke="#FFFF00" stroke-width="2" marker-end="url(#arg)"/><rect x="475" y="100" width="130" height="76" rx="8" fill="#9A72AC" fill-opacity="0.15" stroke="#9A72AC"/><text class="node-label" x="540" y="128" text-anchor="middle">f-string</text><text class="node-sub" x="540" y="148" text-anchor="middle">+ project-plans CSS</text><path class="flow-arrow" d="M605 125 C650 80, 700 55, 740 55" stroke="#FFFF00" stroke-width="2" marker-end="url(#arg)"/><path class="flow-arrow" d="M605 152 C650 195, 700 220, 740 220" stroke="#FFFF00" stroke-width="2" marker-end="url(#arg)"/><rect x="740" y="30" width="145" height="48" rx="8" fill="#83C167" fill-opacity="0.2" stroke="#83C167" stroke-width="2"/><text class="node-label" x="812" y="58" text-anchor="middle">interview-notes.html</text><rect x="740" y="195" width="145" height="48" rx="8" fill="#58C4DD" fill-opacity="0.2" stroke="#58C4DD" stroke-width="2"/><text class="node-label" x="812" y="223" text-anchor="middle">gen-interview-notes.html</text><text font-family="IBM Plex Sans,sans-serif" font-size="10" fill="rgba(255,255,255,0.4)" x="450" y="275" text-anchor="middle">49 questions · 8 categories · Anthropic solo AI/ML engineer focus</text></svg>'''

PITCH = """Employees need policy answers they can trust and verify — not confident hallucinations. As a solo AI/ML engineer, I built an evaluation-first RAG system across two corpora (308 chunks): hybrid BM25+dense retrieval, rerank, grounded generation, faithfulness guard, code validation, and citation-filtered UI. On the 15-case policy benchmark I recovered relevancy from 0.40 to 0.747 (+87%) while raising context precision to 0.80. On the guidebook track I fixed cross-corpus bleed, passed the full 35-case relevancy gate at 0.700 (run 164848), and raised enumeration relevancy to 0.84. The key insight for alignment work: maximizing faithfulness alone collapsed relevancy to 0.42; I measured that Pareto frontier explicitly instead of optimizing one headline metric."""

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="theme-color" content="#e4eaf4">
  <title>Interview Notes — Company Policy RAG · Anthropic</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{base_css}
{CSS_EXTRA}
  </style>
</head>
<body>
  <div class="liquid-bg" aria-hidden="true"><div class="blob blob-1"></div><div class="blob blob-2"></div><div class="blob blob-3"></div><div class="blob blob-4"></div></div>
  <header class="mobile-header">
    <button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation"><svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
    <span class="mobile-header-title">Interview Notes · Anthropic</span>
  </header>
  <div class="toc-backdrop" id="toc-backdrop"></div>
  <div class="layout">
    <aside class="toc" id="toc-panel">
      <button class="toc-close" id="toc-close" type="button" aria-label="Close">&times;</button>
      <div class="toc-title">Interview Prep</div>
      <nav>
        <div class="toc-section">Core</div>
        <a href="#s1">1. 60-Second Pitch</a>
        <a href="#s2">2. Problem &amp; Success</a>
        <a href="#s3">3. Architecture</a>
        <a href="#s4">4. Components</a>
        <div class="toc-section">Depth</div>
        <a href="#s5">5. ML / Eval Rigor</a>
        <a href="#s6">6. Production</a>
        <a href="#s7">7. Agentic Workflow</a>
        <a href="#s8">8. Challenges</a>
        <a href="#s9">9. Results</a>
        <a href="#s10">10. Future</a>
        <div class="toc-section">Interview</div>
        <a href="#s11">11. Question Bank</a>
        <a href="#s12">12. Whiteboard</a>
        <a href="#s13">13. Positioning</a>
        <a href="#s14">14. Gaps</a>
        <a href="#s15">15. Follow-up Answers</a>
      </nav>
    </aside>
    <main>
      <header class="hero" id="s1">
        <span class="anthropic-badge">Anthropic Target · Solo AI/ML Engineer</span>
        <h1>Company Policy RAG<br>Topper-Level Interview Notes</h1>
        <p class="subtitle">Production grounded Q&amp;A for HR/legal PDFs — eval-driven, citation-trustworthy, alignment-aware.</p>
        <div class="pitch-box" id="pitch-text">{PITCH}<em>Speak naturally in ~60 seconds. Do not say "solved hallucinations" — say reduced unsupported claims on golden set.</em></div>
        <div class="btn-row">
          <button class="btn" id="copy-pitch" type="button">Copy 60s Pitch</button>
        </div>
        <div class="hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric"><strong>0.700</strong> Guidebook Rel</span>
          <span class="stat-pill metric"><strong>0.84</strong> Enum Rel</span>
          <span class="stat-pill"><strong>308</strong> Chunks</span>
          <span class="stat-pill"><strong>180</strong> Tests</span>
          <span class="stat-pill"><strong>60</strong> Golden Cases</span>
        </div>
      </header>

      <section id="s2">
        <h2>2. Problem Definition &amp; Success Criteria</h2>
        <p>Employees ask natural-language questions about handbook policies. Answers must be <strong>faithful</strong> (no invented benefits), <strong>relevant</strong> (actually answer the question), and <strong>citation-precise</strong> (sources match what grounded the answer). These three fail <em>independently</em>.</p>
        <div class="card">
          <h3>Business context</h3>
          <ul class="bullets">
            <li>Internal HR / policy Q&amp;A — wrong answer → real employee harm (e.g. sick-leave days)</li>
            <li>Legal-adjacent: abstention is correct when topic absent; hallucination is never acceptable</li>
            <li>Solo build: local stack (Ollama), no cloud API dependency, Docker optional</li>
          </ul>
          <h3>Success criteria (defined upfront)</h3>
          <div class="table-wrap"><table class="trade-table">
            <tr><th>Metric</th><th>Target</th><th>Best achieved</th><th>Run</th></tr>
            <tr><td>Answer Relevancy</td><td>≥ 0.75</td><td class="high-score">0.747</td><td>104356</td></tr>
            <tr><td>Faithfulness (balanced)</td><td>≥ 0.90</td><td>0.807</td><td>104356</td></tr>
            <tr><td>Context Precision</td><td>&gt; 0.50</td><td class="high-score">0.80</td><td>104356</td></tr>
            <tr><td>Hit Rate</td><td>&gt; 0.85</td><td>0.867</td><td>104356</td></tr>
          </table></div>
          <p class="soundbite">"A system can be faithful but useless, or relevant but untrustworthy because sources lie."</p>
        </div>
      </section>

      <section id="s3">
        <h2>3. End-to-End Architecture</h2>
        <div class="diagram-card b3b">
          <div class="diagram-title">Pipeline<span>RAG + Trust Layer</span></div>
          <div class="diagram-viewport tall">{SVG_PIPELINE}</div>
        </div>
        <div class="diagram-card b3b">
          <div class="diagram-title">Alignment Trade-off<span>Strict vs Balanced</span></div>
          <p class="scroll-hint">← Swipe diagram →</p>
          <div class="diagram-viewport wide">{SVG_TRADEOFF}</div>
        </div>
        <div class="diagram-card b3b">
          <div class="diagram-title">Eval Loop<span>Measure Before Ship</span></div>
          <p class="scroll-hint">← Swipe diagram →</p>
          <div class="diagram-viewport wide">{SVG_EVAL}</div>
        </div>
      </section>

      <section id="s4">
        <h2>4. Component Deep Dive</h2>
        <article class="card" id="comp-indexing">
          <h3>Indexing — <code>src/indexing.py</code></h3>
          <p><strong>Purpose:</strong> PDF → section-enriched chunks → Chroma with incremental SHA-256 hashing.</p>
          <ul class="bullets">
            <li><code>build_index()</code>, <code>probe_chroma_index()</code>, <code>enrich_nodes_with_sections()</code></li>
            <li>640/64 tokens; metadata: section_path, page_number, file_hash</li>
            <li><strong>Alt considered:</strong> fixed char splits — rejected (mid-clause legal splits)</li>
          </ul>
        </article>
        <article class="card">
          <h3>Retrieval — <code>src/retriever.py</code> + <code>postprocessors.py</code></h3>
          <p><strong>Purpose:</strong> Hybrid BM25+dense RRF → k=30 → bge-reranker-large → top 6 → 40% score filter. Corpus scope via <code>retrieval_scope.py</code>.</p>
          <ul class="bullets">
            <li><code>_PostprocessingRetriever</code> fixes LlamaIndex postprocessor gap</li>
            <li><code>hybrid_retrieval.py</code> + <code>bm25_index.py</code> (ENABLE_HYBRID_BM25=true)</li>
            <li><strong>Trade-off:</strong> +51% precision on policy baseline, +rerank latency on CPU</li>
          </ul>
        </article>
        <article class="card">
          <h3>Query processing — <code>src/query_processing.py</code></h3>
          <p>Rule expansion + LLM rewrite (no HyDE). Fail-open. ~200–800ms.</p>
        </article>
        <article class="card">
          <h3>Generation — <code>src/generation.py</code></h3>
          <p><code>GroundedCompactAndRefine</code> → <code>normalize_balanced_answer()</code> → <code>apply_faithfulness_guard()</code> → <code>code_validation.py</code>. Balanced reject keeps answer.</p>
        </article>
        <article class="card">
          <h3>Prompts — <code>src/prompts.py</code></h3>
          <p>strict / balanced / v2_balanced. Mandatory <code>[Source N]</code>. 10 focused few-shots.</p>
        </article>
        <article class="card">
          <h3>Citations — <code>src/citations.py</code></h3>
          <p><code>ContextVar</code> turn tracking. Tag-first <code>select_citations_for_answer()</code>. Never parallel UI retrieval.</p>
        </article>
        <article class="card">
          <h3>Agent — <code>src/agent.py</code></h3>
          <p>ReAct 0.14 <code>agent.run()</code>. <code>policy_search</code> tool = full pipeline. <code>AgentTurnResult</code>.</p>
        </article>
        <article class="card">
          <h3>Evaluation — <code>src/evaluation.py</code></h3>
          <p>60 golden cases (policy + guidebook + subsets). Retrieval + LLM judge. <code>guard_modified</code> + code validation traces. 13 policy runs + Track A guidebook runs in JSON log.</p>
        </article>
      </section>

      <section id="s5">
        <h2>5. ML Pipeline &amp; Modeling Rigor</h2>
        <div class="card">
          <p><strong>Honest framing:</strong> This is not a fine-tuning project. Models are off-the-shelf via Ollama + sentence-transformers reranker.</p>
          <h4>Data</h4>
          <ul class="bullets">
            <li>Dual corpus: 80 policy + 228 guidebook chunks → 308 total; golden v2: 60 cases</li>
            <li>Guidebook mix: 12 factual, 6 pattern, 6 workflow, 5 enumeration, 4 code, 2 edge_case</li>
            <li><code>data/eval/golden_dataset.json</code> + <code>golden_dataset_guidebook.json</code></li>
          </ul>
          <h4>Experimentation (policy + Track A runs)</h4>
          <ul class="bullets">
            <li>Policy: 13 runs — guard, few-shots, TOP_N=6, normalize_balanced_answer → 104356</li>
            <li>Guidebook: re-index, code validation tuning (143246), corpus scope, enumeration (164848 rel 0.700)</li>
          </ul>
          <h4>Eval metrics</h4>
          <div class="table-wrap"><table>
            <tr><th>Metric</th><th>Formula / method</th></tr>
            <tr><td>Hit rate</td><td>Any chunk matches relevant_sections keywords</td></tr>
            <tr><td>Context precision</td><td>relevant_retrieved / total_retrieved</td></tr>
            <tr><td>Context recall</td><td>matched keywords / len(relevant_sections)</td></tr>
            <tr><td>Faithfulness</td><td>LLM judge 0–1 vs formatted context</td></tr>
            <tr><td>Answer relevancy</td><td>Context-aware judge with domain rubric</td></tr>
          </table></div>
        </div>
      </section>

      <section id="s6">
        <h2>6. Productionization &amp; MLOps</h2>
        <div class="card">
          <ul class="bullets">
            <li><strong>Config:</strong> Pydantic <code>Settings</code> in <code>config.py</code> — 12-factor .env</li>
            <li><strong>Serving:</strong> Streamlit :8501; Docker Compose + host Ollama</li>
            <li><strong>Reliability:</strong> fail-open rewrite, reranker graceful degradation, probe_chroma_index(), citation score fallback capped</li>
            <li><strong>Observability:</strong> app.log, citation pipeline stages, eval JSON append-only</li>
            <li><strong>Tests:</strong> 180 pytest across generation, citations, prompts, eval, chroma, code validation, retrieval scope</li>
            <li><strong>Gaps (say honestly):</strong> no CI/CD (Phase 4 deferred), no drift monitoring, no live-traffic p50/p95 (golden-set benchmark: 53.5s e2e p50), CPU-only default</li>
          </ul>
        </div>
      </section>

      <section id="s7">
        <h2>7. Agentic / Workflow Engineering</h2>
        <div class="card">
          <ul class="bullets">
            <li>ReAct agent — single <code>policy_search</code> tool (no forked retrieval)</li>
            <li><code>ChatMemoryBuffer</code>: 5 turns, 3000 tokens; <code>build_retrieval_query()</code> expands history</li>
            <li>Validation layers: prompts → normalize → guard → citation tags</li>
            <li><code>begin_citation_turn()</code> / <code>ContextVar</code> for per-turn source tracking</li>
            <li>max_iterations=8; LlamaIndex 0.14 migration from deprecated API</li>
          </ul>
        </div>
      </section>

      <section id="s8">
        <h2>8. Challenges, Failures &amp; Learnings</h2>
        <div class="timeline">
          <div class="timeline-item"><h4>Regression 091001 — Relevancy 0.40</h4><p>Guard replaced good answers with abstention. Fixed: balanced reject → log → keep answer. Recovery to 0.747.</p></div>
          <div class="timeline-item"><h4>Strict mode trap — Faith 1.00 / Relv 0.42</h4><p>Maximizing faithfulness alone unusable for HR. Named Pareto frontier explicitly.</p></div>
          <div class="timeline-item"><h4>Citation trust bug</h4><p>Parallel UI retrieval showed Holidays for sick-leave. Fixed SourceTracking + mandatory [Source N].</p></div>
          <div class="timeline-item"><h4>Chroma false negative</h4><p>Streamlit Chunks: 0 vs 308 actual. Fixed cache reset + NoOpProductTelemetry.</p></div>
          <div class="timeline-item"><h4>Cross-corpus bleed</h4><p>Guidebook questions retrieved handbook chunks. Fixed retrieval_scope.py source_file filter.</p></div>
          <div class="timeline-item"><h4>Code validation 0% pass</h4><p>False-positive fallbacks on valid code. Tuned answer_only trigger → 100% pass (143246).</p></div>
          <div class="timeline-item"><h4>Enumeration incompleteness</h4><p>six_building_blocks rel 0.0. Multi-query + section-diverse rerank → full guidebook rel 0.700 (164848).</p></div>
        </div>
        <p class="soundbite">"The project's memory is logs/evaluation_results.json — ship nothing without a trend line."</p>
      </section>

      <section id="s9">
        <h2>9. Quantified Results</h2>
        <p class="scroll-hint">← Swipe table →</p>
        <div class="table-wrap"><table>
          <tr><th>Run</th><th>Notes</th><th>CtxPrec</th><th>Faith</th><th>Relv</th></tr>
          <tr><td>052233</td><td>Baseline</td><td>0.53</td><td>0.84</td><td>0.71</td></tr>
          <tr><td>073816</td><td>Strict</td><td>0.82</td><td class="high-score">1.00</td><td class="low-score">0.42</td></tr>
          <tr><td>091001</td><td>Regression</td><td>0.82</td><td>0.94</td><td class="low-score">0.40</td></tr>
          <tr class="highlight-row"><td>104356</td><td>Best balanced (policy)</td><td class="high-score">0.80</td><td>0.807</td><td class="high-score">0.747</td></tr>
          <tr><td>152255</td><td>Guidebook post-tuning (35 cases)</td><td>0.657</td><td>0.564</td><td>0.629</td></tr>
          <tr><td>160052</td><td>Enumeration subset (5 cases)</td><td class="high-score">1.00</td><td>—</td><td class="high-score">0.84</td></tr>
          <tr class="highlight-row"><td>164848</td><td>Full guidebook post-enumeration</td><td>0.771</td><td>0.629</td><td class="high-score">0.700</td></tr>
        </table></div>
        <p>Policy per-case: sick_leave relevancy 0.0→0.90. Guidebook: 0.629→0.700 on full 35-case run; enumeration bucket rel 0.84.</p>
      </section>

      <section id="s10">
        <h2>10. Future Improvements</h2>
        <ul class="bullets">
          <li>Phase 4 CI: pytest + stratified eval smoke (guidebook gate 0.700 passed on 164848)</li>
          <li>Faithfulness ≥0.90 without relevancy loss — tighter generation, not more abstention</li>
          <li>Independent Claude-as-judge; per-user ACL metadata filters; GPU latency path</li>
        </ul>
      </section>

      <section id="s11">
        <h2>11. Interview Question Bank</h2>
        <p>{len(QUESTIONS)} questions · Anthropic-weighted · expand each before mock interview.</p>
        <div class="q-bank-controls btn-row">
          <button class="btn" id="expand-all" type="button">Expand All</button>
          <button class="btn" id="collapse-all" type="button">Collapse All</button>
        </div>
        <div class="cat-label">Opening &amp; Behavioral</div>
        {QB_OPENING}
        <div class="cat-label">Data &amp; Retrieval</div>
        {QB_RETRIEVAL}
        <div class="cat-label">Prompts &amp; Grounding</div>
        {QB_PROMPTS}
        <div class="cat-label">Architecture &amp; System Design</div>
        {QB_ARCHITECTURE}
        <div class="cat-label">Production &amp; MLOps</div>
        {QB_PRODUCTION}
        <div class="cat-label">Agentic / Workflow</div>
        {QB_AGENTIC}
        <div class="cat-label">Trade-offs</div>
        {QB_TRADEOFFS}
        <div class="cat-label">Behavioral &amp; Impact</div>
        {QB_BEHAVIORAL}
      </section>

      <section id="s12">
        <h2>12. Whiteboard &amp; Communication Strategy</h2>
        <div class="comp-grid">
          <div class="comp-card"><h4>Diagram 1: Pipeline</h4><p>Ingest → Retrieve → Generate → Deliver. Label config.py and eval hook.</p></div>
          <div class="comp-card"><h4>Diagram 2: Trade-off fork</h4><p>Strict vs Balanced with faith/relv numbers. Anthropic analog: helpful vs harmless.</p></div>
          <div class="comp-card"><h4>Diagram 3: Eval loop</h4><p>Change → 60 cases (stratified subsets) → judge → JSON log → regression catch.</p></div>
        </div>
        <h3>Soundbites</h3>
        <div class="table-wrap"><table>
          <tr><th>Phrase</th><th>When to use</th></tr>
          <tr><td>"Three metrics fail independently"</td><td>Problem framing</td></tr>
          <tr><td>"Name the Pareto frontier"</td><td>Faithfulness vs relevancy</td></tr>
          <tr><td>"Wrong sources worse than fewer sources"</td><td>Citation trust</td></tr>
          <tr><td>"Measure before optimize"</td><td>Eval discipline</td></tr>
          <tr><td>"Workflow engineering beat model swapping"</td><td>Architecture philosophy</td></tr>
        </table></div>
        <h3>Anthropic-specific emphasis</h3>
        <ul class="bullets">
          <li>Lead with eval harness + honest trade-offs, not "I built a chatbot"</li>
          <li>Connect to alignment: abstention calibration, verifiable outputs, regression detection</li>
          <li>Acknowledge judge limitations (same model family) — propose Claude-as-judge as improvement</li>
        </ul>
      </section>

      <section id="s13">
        <h2>13. Positioning &amp; Red Flags</h2>
        <div class="card">
          <h3>Frame as</h3>
          <p>Solo production RAG with alignment-aware eval — workflow-first grounding system, not a notebook demo.</p>
          <h3>Weak areas (address honestly)</h3>
          <ul class="bullets">
            <li>Guidebook faithfulness 0.629 vs 0.90 target; code-query rel 0.525 (run 164848)</li>
            <li>Policy faithfulness 0.807 vs 0.90 target on best balanced run</li>
            <li>7B local model limits; CPU e2e p50 53.5s; no production traffic metrics</li>
            <li>LLM judge not independent from generator; Phase 4 CI not wired yet</li>
          </ul>
          <h3 class="red-flag">Red flags to avoid</h3>
          <ul class="bullets">
            <li class="red-flag">Claiming "achieved 0.75 relevancy" — say 0.747, within 0.003</li>
            <li class="red-flag">"Solved hallucinations" — say reduced unsupported claims on eval set</li>
            <li class="red-flag">Hiding strict-mode relevancy collapse</li>
            <li class="red-flag">Treating faithfulness and relevancy as one metric</li>
          </ul>
        </div>
      </section>

      <section id="s14">
        <h2>14. Missing Information (strengthen your story)</h2>
        <div class="card">
          <ul class="bullets">
            <li><strong>Resolved:</strong> 5-case human overlap — faith κ@0.5 1.0, relv κ@0.5 0.0 (logs/human_judge_agreement.json)</li>
            <li><strong>Resolved:</strong> Git-derived timeline — ~0.5 weeks span, 13 eval runs in 5.4h (docs/project_timeline.json)</li>
            <li><strong>Resolved:</strong> Measured p50/p95 — e2e 53.5s / 58.8s on 5 golden cases (logs/latency_benchmark.json)</li>
            <li><strong>Resolved:</strong> Hybrid BM25 shipped; corpus scoping; code validation 100% pass (143246); guidebook rel gate 0.700 (164848)</li>
            <li><strong>Remaining:</strong> Independent judge (Claude API); Phase 4 CI wiring; faithfulness recovery on guidebook</li>
          </ul>
        </div>
      </section>

      <section id="s15">
        <h2>15. Follow-up Answer Bank</h2>
        <p>98 mature answers · Anthropic-caliber · click any follow-up in <a href="#s11">Section 11</a> to jump here. Highlighted card = your target answer.</p>
        <div class="btn-row">
          <button class="btn" id="expand-fu-groups" type="button">Expand All Answers</button>
          <button class="btn" id="collapse-fu-groups" type="button">Collapse Groups</button>
        </div>
        {FOLLOWUP_BANK}
      </section>

      <footer>
        <p><strong>Company Policy RAG</strong> — Interview Notes for Anthropic<br>
        <a href="project-plans.html">Engineering Plans</a> · <a href="gen-interview-notes.html">Generator Docs</a> · <a href="../README3.md">README3</a> · <a href="https://github.com/SoubhagyaJain/Rag-chatbot">GitHub</a></p>
      </footer>
    </main>
  </div>
  <script>
    (function(){{
      var t=document.getElementById('menu-toggle'),c=document.getElementById('toc-close'),p=document.getElementById('toc-panel'),b=document.getElementById('toc-backdrop');
      function openM(){{p.classList.add('open');b.classList.add('visible');document.body.classList.add('menu-open');t.setAttribute('aria-expanded','true');}}
      function closeM(){{p.classList.remove('open');b.classList.remove('visible');document.body.classList.remove('menu-open');t.setAttribute('aria-expanded','false');}}
      if(t)t.onclick=function(){{p.classList.contains('open')?closeM():openM();}};
      if(c)c.onclick=closeM; if(b)b.onclick=closeM;
      document.querySelectorAll('#toc-panel nav a').forEach(function(a){{a.onclick=function(){{if(window.matchMedia('(max-width:900px)').matches)closeM();}};}});
      document.getElementById('copy-pitch').onclick=function(){{
        navigator.clipboard.writeText(document.getElementById('pitch-text').innerText.split('Speak naturally')[0].trim());
        this.textContent='Copied!'; var s=this; setTimeout(function(){{s.textContent='Copy 60s Pitch';}},2000);
      }};
      document.getElementById('expand-all').onclick=function(){{document.querySelectorAll('.q-card').forEach(function(d){{d.open=true;}});}};
      document.getElementById('collapse-all').onclick=function(){{document.querySelectorAll('.q-card').forEach(function(d){{d.open=false;}});}};
      document.querySelectorAll('.fu-link').forEach(function(a){{
        a.onclick=function(){{
          var el=document.querySelector(this.getAttribute('href'));
          if(el){{setTimeout(function(){{el.scrollIntoView({{behavior:'smooth',block:'start'}});}},50);}}
        }};
      }});
      var expandFu=document.getElementById('expand-fu-groups');
      if(expandFu)expandFu.onclick=function(){{document.querySelectorAll('.fu-group').forEach(function(g){{g.classList.add('fu-open');}});}};
      var collapseFu=document.getElementById('collapse-fu-groups');
      if(collapseFu)collapseFu.onclick=function(){{document.querySelectorAll('.fu-group').forEach(function(g){{g.classList.remove('fu-open');}});}};
    }})();
  </script>
</body>
</html>'''

OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT} ({len(html)} chars, {len(QUESTIONS)} questions)")

gen_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="theme-color" content="#e4eaf4">
  <title>Generator Docs — Interview Notes · Anthropic</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{base_css}
{CSS_GEN}
  </style>
</head>
<body>
  <div class="liquid-bg" aria-hidden="true"><div class="blob blob-1"></div><div class="blob blob-2"></div><div class="blob blob-3"></div><div class="blob blob-4"></div></div>
  <header class="mobile-header">
    <button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation"><svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
    <span class="mobile-header-title">Generator Docs</span>
  </header>
  <div class="toc-backdrop" id="toc-backdrop"></div>
  <div class="layout">
    <aside class="toc" id="toc-panel">
      <button class="toc-close" id="toc-close" type="button" aria-label="Close">&times;</button>
      <div class="toc-title">Generator</div>
      <nav>
        <a href="#g1">1. Overview</a>
        <a href="#g2">2. Pipeline</a>
        <a href="#g3">3. Regenerate</a>
        <a href="#g4">4. Categories</a>
        <a href="#g5">5. Source Bank</a>
        <a href="#g6">6. Code Reference</a>
        <a href="#g7">7. Gotchas</a>
      </nav>
    </aside>
    <main>
      <header class="hero gen-hero" id="g1">
        <span class="anthropic-badge">Source of Truth · Solo AI/ML Engineer</span>
        <h1>Interview Notes Generator</h1>
        <p class="subtitle">Python source → premium HTML/CSS · {len(QUESTIONS)} Anthropic-weighted questions · liquid glass theme from project-plans.</p>
        <div class="file-grid">
          <div class="file-pill"><strong>Input</strong><code>docs/_gen_interview_notes.py</code></div>
          <div class="file-pill out"><strong>Output A</strong><a href="interview-notes.html">interview-notes.html</a></div>
          <div class="file-pill out"><strong>Output B</strong><a href="gen-interview-notes.html">gen-interview-notes.html</a></div>
          <div class="file-pill"><strong>CSS source</strong><code>project-plans.html</code></div>
        </div>
      </header>

      <section id="g2">
        <h2>2. Generation Pipeline</h2>
        <p class="scroll-hint">← Swipe diagram →</p>
        <div class="diagram-card b3b">
          <div class="diagram-viewport wide">{SVG_GEN}</div>
        </div>
        <div class="step-row">
          <span class="step-chip">add_qs tuples</span><span class="step-arrow">→</span>
          <span class="step-chip">q() accordion HTML</span><span class="step-arrow">→</span>
          <span class="step-chip">qb_for() by category</span><span class="step-arrow">→</span>
          <span class="step-chip">f-string template</span><span class="step-arrow">→</span>
          <span class="step-chip">write both HTML files</span>
        </div>
      </section>

      <section id="g3">
        <h2>3. How to Regenerate</h2>
        <div class="card">
          <p>From the <code>company_policy_rag</code> repo root:</p>
          <div class="cmd-box"><code>python docs/_gen_interview_notes.py</code></div>
          <ul class="bullets">
            <li>Edits question tuples in <code>add_qs("category", [...])</code> blocks</li>
            <li>Re-reads base CSS from <code>project-plans.html</code> automatically</li>
            <li>Overwrites <code>interview-notes.html</code> and <code>gen-interview-notes.html</code></li>
          </ul>
        </div>
      </section>

      <section id="g4">
        <h2>4. Question Categories</h2>
        <p class="scroll-hint">← Swipe table →</p>
        <div class="table-wrap"><table class="num-table">
          <tr><th>Label</th><th>Key</th><th>Count</th></tr>
          {CAT_STATS_ROWS}
          <tr class="highlight-row"><td colspan="2"><strong>Total</strong></td><td class="num">{len(QUESTIONS)}</td></tr>
        </table></div>
      </section>

      <section id="g5">
        <h2>5. Question Source Bank</h2>
        <p>Raw tuple data behind section 11 — edit these in Python, then regenerate.</p>
        <div class="btn-row">
          <button class="btn" id="expand-src" type="button">Expand All</button>
          <button class="btn" id="collapse-src" type="button">Collapse All</button>
        </div>
        {SOURCE_BANK}
      </section>

      <section id="g6">
        <h2>6. Core Generator Functions</h2>
        <div class="code-panel">{CODE_SNIPPET}</div>
      </section>

      <section id="g7">
        <h2>7. F-String Gotchas</h2>
        <div class="warn-box">
          <strong>Never break the f-string mid-template.</strong> A line like
          <code>&lt;p&gt;''' + str(len(QUESTIONS)) + ''' questions&lt;/p&gt;</code>
          closes the f-string early — everything after renders as literal <code>{{QB_OPENING}}</code> placeholders.
          Use <code>{{len(QUESTIONS)}}</code> inside the f-string instead.
        </div>
        <ul class="bullets">
          <li>JavaScript in the template must use doubled braces: <code>{{{{</code> and <code>}}}}</code></li>
          <li>Pre-build <code>QB_*</code> variables before the template — do not inline <code>join()</code> inside f-string braces</li>
          <li><code>add_qs</code> tuples have 6 fields; pass <code>item[0]..item[4]</code> to <code>q()</code> — not <code>q(*item, cat=cat)</code></li>
        </ul>
      </section>

      <footer>
        <p><strong>Company Policy RAG</strong> — Generator Documentation<br>
        <a href="interview-notes.html">Interview Notes</a> · <a href="project-plans.html">Engineering Plans</a> · <a href="https://github.com/SoubhagyaJain/Rag-chatbot">GitHub</a></p>
      </footer>
    </main>
  </div>
  <script>
    (function(){{
      var t=document.getElementById('menu-toggle'),c=document.getElementById('toc-close'),p=document.getElementById('toc-panel'),b=document.getElementById('toc-backdrop');
      function openM(){{p.classList.add('open');b.classList.add('visible');document.body.classList.add('menu-open');t.setAttribute('aria-expanded','true');}}
      function closeM(){{p.classList.remove('open');b.classList.remove('visible');document.body.classList.remove('menu-open');t.setAttribute('aria-expanded','false');}}
      if(t)t.onclick=function(){{p.classList.contains('open')?closeM():openM();}};
      if(c)c.onclick=closeM; if(b)b.onclick=closeM;
      document.querySelectorAll('#toc-panel nav a').forEach(function(a){{a.onclick=function(){{if(window.matchMedia('(max-width:900px)').matches)closeM();}};}});
      document.getElementById('expand-src').onclick=function(){{document.querySelectorAll('.source-card').forEach(function(d){{d.open=true;}});}};
      document.getElementById('collapse-src').onclick=function(){{document.querySelectorAll('.source-card').forEach(function(d){{d.open=false;}});}};
    }})();
  </script>
</body>
</html>'''

OUT_GEN.write_text(gen_html, encoding="utf-8")
print(f"Wrote {OUT_GEN} ({len(gen_html)} chars, source bank {len(RAW_QUESTIONS)} items)")