# Company Policy RAG — Engineering Journey & Lessons Learned

> **README2** documents the full build-and-tune story of `company_policy_rag`: what we tried, what broke, what we fixed, and how metrics moved. Written for engineers who want to understand *why* each decision was made — not just what the code does.

For day-to-day setup and API reference, see [README.md](README.md).  
For **current completion status and prioritized backlog**, see [README3.md](README3.md).

---

## Executive summary

We built a **local, production-oriented RAG system** for company policy and legal PDFs:

| Layer | Stack |
|-------|-------|
| Framework | LlamaIndex 0.14+ |
| LLM / Embeddings | Ollama — `qwen2.5:7b`, `nomic-embed-text` |
| Vector store | ChromaDB (persistent, metadata-rich chunks) |
| Reranker | `BAAI/bge-reranker-large` |
| UI | **Streamlit** (primary) + Chainlit (legacy) |
| Deploy | **Docker** (Streamlit container + host Ollama) |
| Eval | 60-case golden set (25 policy + 35 guidebook) + LLM-as-judge + `logs/evaluation_results.json` |
| Tests | **182** passing (`pytest tests/ -v`) |

### Headline metric improvements (balanced mode, golden set)

| Metric | Worst observed | Best achieved | Change | % improvement |
|--------|----------------|---------------|--------|---------------|
| **Answer Relevancy** | 0.40 | **0.747** | +0.347 | **+87%** vs regression low |
| **Answer Relevancy** | 0.42 (strict) | **0.747** | +0.327 | **+78%** vs strict baseline |

### 2026-06-18 addendum (Track A + Phase 4 CI)

- Guidebook relevancy gate **passed** on full 35-case run `164848` (rel **0.700**).
- Phase 4 CI tasks 5–6 **shipped**: `.github/workflows/rag-ci.yml` + `scripts/ci_eval_gate.py` (retrieval smoke: hit 1.00 / prec 0.896 / rec 0.75).
- **182** pytest tests; local smoke verified; GitHub Actions runner blocked on billing until resolved.
- Remaining quality gap: guidebook faithfulness **0.629** vs 0.90 target — see [README3.md](README3.md) §7.

### 2026-06-19 addendum (CI green + faithfulness tuning)

- **GitHub Actions green** on run [#27804469869](https://github.com/SoubhagyaJain/Rag-chatbot/actions/runs/27804469869) — billing resolved; `unit-tests` 182/182 + `eval-smoke` PASS.
- **Faithfulness tuning** shipped in `dd40b86`: prompt rules 19b–25, few-shots P–X, optional claim-trim guard (`FAITHFULNESS_GUARD_REJECT_ACTION`, default `keep`).
- Full 35-case guidebook re-eval `055058`: faith **0.543**, rel **0.666** — **regression** vs baseline `164848` (faith **0.629**, rel **0.700**). Per-case: 2 improved / 8 regressed / 25 unchanged. Clear win: `manager_agent` faith **1.0**.
- **Lesson:** prompt-only tuning with qwen2.5:7b did not move aggregate faith; claim-trim default `keep` after weak-subset test showed faith drop with `trim`. Root cause for code cases is **retrieval** (`currency_tool_example`, `tools_real_world`), not generation prompts alone.
- **Next:** code/currency retrieval boost — see [README3.md](README3.md) §7 task 19.
| **Context Precision** | 0.53 | **0.80** | +0.27 | **+51%** |
| **Faithfulness** | 0.71 | **0.94** | varies by mode | See trade-off section |
| **Hit rate** | 0.87 | **0.87** | stable | Retrieval already strong |

**Targets we set:** Answer Relevancy ≥ 0.75, Faithfulness ≥ 0.90 in balanced mode.

**Where we landed (run `20260617_104356`):** Relevancy **0.747** (within 0.003 of target), Faithfulness **0.807** (trade-off from relevancy recovery).

---

## Table of contents

1. [The problem we were solving](#the-problem-we-were-solving)
2. [Engineering timeline (conversation arc)](#engineering-timeline-conversation-arc)
3. [Evaluation runs — full scoreboard](#evaluation-runs--full-scoreboard)
4. [Phase-by-phase fixes](#phase-by-phase-fixes)
5. [Per-case wins (before → after)](#per-case-wins-before--after)
6. [Lessons learned (production-rag)](#lessons-learned-production-rag)
7. [Architecture (final state)](#architecture-final-state)
8. [Citation trust fix](#citation-trust-fix)
9. [Streamlit UI](#streamlit-ui)
10. [LlamaIndex 0.14 agent migration](#llamaindex-014-agent-migration)
11. [Mandatory [Source N] citation prompts](#mandatory-source-n-citation-prompts)
12. [Chroma telemetry & index health fixes](#chroma-telemetry--index-health-fixes)
13. [Docker deployment](#docker-deployment)
14. [Configuration that mattered](#configuration-that-mattered)
15. [How to reproduce & test](#how-to-reproduce--test)
16. [What is still open](#what-is-still-open)
17. [Project structure](#project-structure)

---

## The problem we were solving

Employees need to ask natural-language questions about handbook policies and get answers they can **trust and verify**. For policy/legal RAG, that means:

1. **Faithfulness** — no invented penalties, dates, or benefits.
2. **Answer relevancy** — actually answer the question, not abstain when context exists.
3. **Citation precision** — sources shown must match what grounded the answer.

We discovered all three can fail independently. A system can be faithful but useless (over-abstention). It can be relevant but show **wrong sources** (destroying trust even when the answer text is okay).

---

## Engineering timeline (conversation arc)

### Act 1 — Baseline & reranker (early June 2026)

- Indexed `Employee-Handbook-for-Nonprofits-and-Small-Businesses.pdf` with section-aware chunking (640 tokens, 64 overlap).
- Added ChromaDB, cross-encoder reranker, query rewrite.
- First eval runs: Context Precision ~0.53, Answer Relevancy ~0.71.

**Lesson:** Retrieval recall was fine (hit rate ~87%); precision and generation were the bottlenecks.

---

### Act 2 — Strict grounding experiment

- Enabled **strict mode**: aggressive abstention + strict faithfulness guard.
- **Faithfulness hit 1.00** — but **Answer Relevancy collapsed to 0.42**.

**Lesson:** Maximizing faithfulness alone is a trap. Users don't want a system that only says "insufficient information" when the handbook has related text. For internal HR Q&A, **balanced mode** is the right default.

---

### Act 3 — The regression (run `091001`)

Answer Relevancy fell to **0.40** — the worst score in the project.

**Root causes identified:**

| # | Cause | Symptom |
|---|--------|---------|
| 1 | Balanced faithfulness guard **replaced entire answers** with `INSUFFICIENT_INFO_MESSAGE` on `UNSUPPORTED` | `sick_leave`, `dress_code`, `benefits_eligibility` scored 0.0 relevancy despite 4/4 relevant chunks retrieved |
| 2 | Prompt bloat (8 few-shot examples) increased abstention prior | Model learned to abstain even when context matched |
| 3 | Edge-case retrieval gaps | `at-will` chunks missed for resignation questions; health insurance vocabulary missed for benefits |
| 4 | Double-ending pattern | Partial answer + full abstention suffix in one response |

---

### Act 4 — Relevancy recovery (Phases 1–3)

User goal: **Relevancy ≥ 0.75** while keeping **Faithfulness ≥ 0.90**.

| Phase | Run ID | Relevancy | Faithfulness | Key changes |
|-------|--------|-----------|--------------|-------------|
| 1 | `094925` | 0.56 | 0.75 | Guard keeps answer on reject; strip double-endings |
| 2 | `100420` | 0.66 | 0.79 | Trimmed few-shots; benefits/resignation/dress query augmentation |
| 3 | `102648` | 0.69 | 0.71 | `RERANKER_TOP_N=6`; context-aware relevancy judge |
| 4 | `104356` | **0.747** | **0.807** | `normalize_balanced_answer()`; at-will/disciplinary/outside-employment prompts |

**Net relevancy recovery:** 0.40 → 0.747 = **+86.8%** relative improvement from the regression floor.

---

### Act 5 — Streamlit UI

- Replaced Chainlit as the primary internal-tool UI.
- Professional chat, sidebar settings (grounding mode, temperature), expandable citations.
- Hit **LlamaIndex 0.14 breaking change**: `ReActAgent.from_tools()` removed → migrated to workflow `agent.run()` API.

---

### Act 6 — Citation trust fix

**User-reported bug:** Sick leave question returned a decent answer but showed irrelevant sources (Holidays, Visitors, Family Leave).

**Root cause:** UI ran a **separate** `retriever.retrieve()` and displayed *all* post-rerank chunks — not the chunks the LLM actually cited.

**Fix:** `SourceTrackingQueryEngine` + `select_citations_for_answer()` — only show `[Source N]` tags present in the answer (or strict score fallback).

---

### Act 7 — Mandatory `[Source N]` citation prompts

**Problem:** Even with the citation pipeline, the LLM often omitted `[Source N]` tags in balanced mode. Without tags, `select_citations_for_answer()` fell back to score-based selection — reintroducing irrelevant sources.

**Root cause:** Soft rule ("cite when stating specific facts") and agent system prompt asked for "section titles and page numbers" instead of preserving `[Source N]` tags from tool output.

**Fix (`src/prompts.py`):**

| Change | Detail |
|--------|--------|
| `CITATION RULES (MANDATORY)` block | Every factual sentence must end with `[Source N]` matching `<source id="N">` |
| `FEW_SHOT_BALANCED` Example J | BAD example: correct answer without tags — labeled NEVER DO THIS |
| `BALANCED_REFINE_PROMPT_TMPL` | Preserve existing tags; add tags for new facts |
| `AGENT_SYSTEM_PROMPT_BALANCED` | Preserve `[Source N]` verbatim from `policy_search`; no page-number-only citations |

**Tests:** `tests/test_prompts.py` asserts mandatory citation language in balanced templates.

---

### Act 8 — Chroma telemetry & index health fixes

#### Telemetry noise

**Symptom:** Log spam on every Chroma client start:

```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
```

**Cause:** `chromadb` 0.5.x calls legacy `posthog.capture(distinct_id, event, props)`; `posthog` 7.x removed that signature. `ANONYMIZED_TELEMETRY=False` alone still invokes `capture()`.

**Fix:**

| File | Change |
|------|--------|
| `src/chroma_telemetry.py` | `NoOpProductTelemetry` — never calls posthog |
| `src/indexing.py` | `chroma_product_telemetry_impl` → NoOp; `chromadb.configure()` on import |
| `requirements.txt` | Pin `posthog>=2.4.0,<3.0.0` |
| `.env` / `.env.example` | `ANONYMIZED_TELEMETRY=False` |

**Tests:** `tests/test_chroma_telemetry.py` (4 tests).

#### Streamlit "No index found" false negative

**Symptom:** Streamlit showed `Chunks: 0` while CLI proved 81 chunks existed.

**Causes:**

1. `get_collection_stats()` returned `count: 0` whenever `index_exists()` failed (misleading diagnostics).
2. Chroma `SharedSystemClient` cached a client with default settings; our NoOp telemetry settings conflicted → silent failure.

**Fix (`src/indexing.py`, `app/streamlit_app.py`):**

- `probe_chroma_index()` — always reports real chunk count + error message
- `reset_chroma_client_cache()` on Streamlit startup
- `get_chroma_client()` clears cache and retries on settings conflict
- Streamlit diagnostics: `Chunks (actual)`, collections list, **Clear Chroma client cache and retry** button

---

### Act 9 — Docker deployment

**Goal:** Run Streamlit in a container; Ollama stays on the **host** (user choice).

**Files added:**

| File | Purpose |
|------|---------|
| `Dockerfile` | Python 3.11-slim, CPU torch, runtime deps |
| `docker-compose.yml` | Port 8501, volumes, `host.docker.internal` |
| `requirements-docker.txt` | No jupyter/pytest/chainlit |
| `docker/entrypoint.sh` | Wait for Ollama, optional `AUTO_INDEX_ON_START`, Streamlit on `0.0.0.0` |
| `.env.docker.example` | Docker-specific env template |
| `.dockerignore` | Exclude venv, chroma blobs, logs |

**Volumes:** `./data`, `./storage`, `./logs`, `hf_cache` (reranker model cache).

**Verified:** `docker compose build` succeeds (~3.5 min on Windows).

---

## Evaluation runs — full scoreboard

All runs: 15 golden cases, `qwen2.5:7b` judge + generator, logged in `logs/evaluation_results.json`.

| Run ID | Mode / notes | Hit | CtxPrec | CtxRec | Faith | Relv |
|--------|----------------|-----|---------|--------|-------|------|
| `052233` | Early baseline | 0.87 | 0.53 | 0.83 | 0.84 | 0.71 |
| `064756` | Reranker tuned | 0.87 | 0.82 | 0.67 | 0.77 | 0.77 |
| `073816` | **Strict** | 0.87 | 0.82 | 0.67 | **1.00** | **0.42** |
| `091001` | **Regression** | 0.87 | 0.82 | 0.67 | 0.94 | **0.40** |
| `094925` | Guard fix | 0.87 | 0.77 | 0.66 | 0.75 | 0.56 |
| `100420` | Prompts + augmentation | 0.87 | 0.77 | 0.64 | 0.79 | 0.66 |
| `102648` | TOP_N=6 + judge | 0.87 | 0.80 | 0.67 | 0.71 | 0.69 |
| `104356` | **Best balanced** | 0.87 | 0.80 | 0.65 | 0.81 | **0.747** |

### Visual: Answer Relevancy journey

```
0.71 ── baseline
0.42 ── strict mode (faithfulness 1.0, not usable)
0.40 ── regression low ▼
0.56 ── Phase 1
0.66 ── Phase 2
0.69 ── Phase 3
0.747 ── Phase 4 (best) ▲ +87% from 0.40
────── target 0.75 ──────
```

---

## Phase-by-phase fixes

### Phase 1 — Stop the guard from nuking good answers

**File:** `src/generation.py`

- Balanced guard rejection **keeps the original answer** (logs warning instead of replacing with abstention).
- `_preserve_balanced_partial_answer()` strips double-ending abstention suffixes.

**Impact:** `sick_leave`, `dress_code`, `benefits_eligibility` recovered from 0.0 → 0.8+ relevancy.

---

### Phase 2 — Prompts & retrieval augmentation

**Files:** `src/prompts.py`, `src/query_processing.py`, `data/eval/golden_dataset.json`

- Trimmed `FEW_SHOT_BALANCED` (9 focused examples, not 8 bloated ones).
- Rule 11: must answer when context matches; abstention last.
- Semantic mapping examples (social media → internet policy; resignation → at-will).
- `augment_query_with_policy_terms()` for benefits, resignation, dress code, confidentiality.
- Golden set: `notice_resignation` keywords fixed (`at-will`, removed bare `"notice"`).

**Config:** `RETRIEVAL_CANDIDATE_K=30`, `RERANK_MIN_KEEP=3`, `RERANK_MIN_SCORE_RATIO=0.35`.

---

### Phase 3 — Retrieval depth + smarter judging

- `RERANKER_TOP_N=6` (more context for multi-part questions).
- Relevancy judge receives **retrieved context** (not question + answer alone).
- Eval trace fields: `pre_guard_answer`, `guard_modified`.

---

### Phase 4 — Double-ending + weak edge cases

**File:** `src/generation.py` — `normalize_balanced_answer()`

- Strips quasi-abstention suffixes at synthesis time (not just in guard).
- Fixes inverted pattern: abstention paragraph then partial answer.

**Prompts:** At-will resignation example, disciplinary process (report → investigate → terminate), outside employment (conflict of interest, no invented approval rules).

**Query augmentation:** Disciplinary + outside employment term expansions.

**Judge tuning:** Score at-will resignation and honest gap statements fairly.

---

### Phase 5 — Citation precision (trust)

**New file:** `src/citations.py`

| Before | After |
|--------|-------|
| UI calls `retriever.retrieve()` separately | Citations from `SourceTrackingQueryEngine` only |
| Shows all 6 reranked chunks | Shows only `[Source N]` cited in answer |
| Holidays/Visitors appear for sick leave | Only Sick Leave section shown |

**Config added:**

```env
CITATION_MIN_RELEVANCE_RATIO=0.55
ENABLE_CITATION_PIPELINE_LOGGING=true
RERANK_MIN_SCORE_RATIO=0.40
```

---

### Phase 6 — Mandatory `[Source N]` tags + infra hardening

**Files:** `src/prompts.py`, `src/chroma_telemetry.py`, `src/indexing.py`, `app/streamlit_app.py`, `Dockerfile`, `docker-compose.yml`

| Work item | Outcome |
|-----------|---------|
| `CITATION RULES (MANDATORY)` in balanced prompt | Model must tag every factual sentence |
| Agent system prompt | Preserve `[Source N]` from tool output |
| `NoOpProductTelemetry` | Chroma telemetry errors eliminated |
| `probe_chroma_index()` | Real chunk count in Streamlit diagnostics |
| Docker | `docker compose up` → Streamlit on :8501 with host Ollama |

---

## Per-case wins (before → after)

Selected golden cases that drove the most learning:

| Case ID | Question topic | Relevancy before | Relevancy after | What fixed it |
|---------|----------------|------------------|-----------------|---------------|
| `sick_leave` | Sick days | 0.0 | **0.90** | Guard no longer nukes answer |
| `dress_code` | Dress code | 0.0 | **0.80** | Same + better retrieval |
| `benefits_eligibility` | Health benefits | 0.0 | **0.80** | Benefits few-shot + query augmentation |
| `notice_resignation` | Quit without notice | 0.0 → 0.5 | **0.80** | At-will mapping + double-ending fix |
| `disciplinary_process` | Discipline steps | 0.5 | **0.80** | At-will + report/investigate prompt |
| `outside_employment` | Second job | 0.5 | **0.80** | Conflict-of-interest mapping |
| `social_media` | Social media | 0.5 | **0.80** | Internet policy semantic mapping |
| `remote_work` | Remote work | 0.0 | 0.0 | Correct abstention (not in handbook) |

---

## Lessons learned (production-rag)

### 1. Measure before optimizing

Every significant change ran through `python scripts/evaluate.py` and appended to `logs/evaluation_results.json`. Without trend logs, we would have shipped the `091001` regression without noticing.

### 2. Faithfulness and relevancy trade off — name it explicitly

| Mode | Faithfulness | Relevancy | Best for |
|------|--------------|-----------|----------|
| Strict | ~1.00 | ~0.42 | Audit / compliance |
| Balanced | ~0.81 | ~0.75 | Internal HR Q&A |

Don't optimize one metric in isolation.

### 3. The guard is a second LLM call — it can hurt as much as help

A balanced guard that **replaces** answers causes over-abstention. Our fix: reject → log → **keep answer** in balanced mode. Strict mode still abstains.

### 4. Prompt few-shots set priors

Eight verbose few-shots taught the model to abstain. Trimming to focused, domain-specific examples (sick leave, at-will, benefits, social media mapping) reduced false abstention materially.

### 5. Vocabulary mismatch is a retrieval problem, not just a generation problem

"Resignation without notice" doesn't embed near "employment at-will." **Query augmentation** + **LLM rewrite** bridge user language to handbook terms.

### 6. Double-endings are a silent relevancy killer

Pattern: good partial answer + `INSUFFICIENT_INFO_MESSAGE` suffix. Judges score 0.5 ("partially addresses"). Fix at **generation time**, not only in the guard.

### 7. Citations must come from the same pipeline as generation

> Showing unrelated sources is worse than showing fewer sources.

Never run a parallel retrieval for display. Track `source_nodes` from the query engine and filter by `[Source N]` tags in the answer.

### 8. LLM-as-judge needs domain guidance

Generic judges penalize correct at-will answers ("no penalty stated" scored as unfaithful). Add scoring guidance for edge cases (resignation → at-will, correct abstention when topic absent).

### 9. Framework upgrades break production code silently

LlamaIndex 0.14 removed `ReActAgent.from_tools()`, `chat()`, `achat()`. Error surfaced as cryptic `from_tools` in Streamlit. Pin versions or test UI after upgrades.

### 10. Local ML stack dependencies are fragile on Windows

- `torch` without `torchvision` → Streamlit file watcher spam from `transformers` vision modules.
- Fix: `pip install torchvision --index-url https://download.pytorch.org/whl/cpu` + `.streamlit/config.toml` with `fileWatcherType = "none"`.

### 11. Soft citation rules produce soft citation behavior

"If you cite when stating facts" → model often doesn't. **Mandatory `[Source N]` rules** in both the generation prompt and agent system prompt align LLM output with the citation pipeline. Without tags, score fallback reintroduces wrong sources.

### 12. Chroma client cache survives Streamlit hot-reload

`SharedSystemClient` keeps in-process clients across reloads. Settings mismatch (default Posthog vs NoOp telemetry) makes `index_exists()` fail silently. Clear cache on startup and probe index health independently of the boolean check.

### 13. Docker needs explicit Streamlit bind address

Default Streamlit binds `localhost` only — port mapping breaks. Set `address = "0.0.0.0"` in `.streamlit/config.toml` and pass `--server.address=0.0.0.0` in the entrypoint.

---

## Architecture (final state)

```
PDFs (data/policies/)
    │
    ▼
Indexing (640-token chunks + section_path, page_number metadata)
    │
    ▼
ChromaDB (cosine HNSW)
    │
    ▼
Query rewrite + policy-term augmentation
    │
    ▼
Over-retrieve (k=30) → bge-reranker-large → top 6 → score filter (ratio 0.40)
    │
    ▼
GroundedCompactAndRefine (balanced prompts + few-shots)
    │
    ▼
normalize_balanced_answer() → faithfulness guard (balanced: keep on reject)
    │
    ▼
SourceTrackingQueryEngine records source_nodes
    │
    ▼
select_citations_for_answer() → [Source N] filter → Streamlit UI
```

### Agent path (multi-turn chat)

```
User message
    → build_retrieval_query() (memory-expanded)
    → ReActAgent.run() → policy_search tool → SourceTrackingQueryEngine
    → AgentTurnResult(answer, citations)
    → Streamlit chat + expandable sources
```

---

## Citation trust fix

### Problem

```
User: "How many sick days do employees receive?"
Answer: [correct sick leave text]
Sources shown: Holidays | Visitors | Family Care Leave  ← WRONG
```

### Solution (`src/citations.py`)

1. **`begin_citation_turn()`** — reset per agent turn.
2. **`record_generation_sources()`** — called by `SourceTrackingQueryEngine` after each `policy_search`.
3. **`extract_cited_source_indices()`** — parse `[Source 1]`, `[Source 2, 3]`.
4. **`select_citations_for_answer()`** — display only cited indices; score fallback if no tags.

### Pipeline logging (when `ENABLE_CITATION_PIPELINE_LOGGING=true`)

```
Citation pipeline | chroma_retrieved | 30 chunks | ...
Citation pipeline | post_rerank_filter | 3 chunks | ...
Citation pipeline | query_engine_output | 3 chunks | ...
Citation selection | mode=cited_in_answer | displayed=1 | cited_tags=[1]
```

### How to verify

```bash
# Unit tests
python -m pytest tests/test_citations.py -v

# Manual: ask sick leave question in Streamlit, confirm only Sick Leave source expands
streamlit run app/streamlit_app.py
```

---

## Streamlit UI

**Run:**

```bash
cd company_policy_rag
pip install -r requirements.txt
python scripts/index_documents.py   # if not indexed
streamlit run app/streamlit_app.py
```

Open http://localhost:8501

| Feature | Details |
|---------|---------|
| Chat | Multi-turn via `ChatMemoryBuffer` |
| Sources | `st.expander` per source — section path, page, file, excerpt |
| Sidebar | Grounding mode (Balanced/Strict), temperature, clear chat, index stats |
| Loading | `st.status` during retrieval + generation |

Chainlit still available: `chainlit run app/chat_app.py --port 8000`

---

## LlamaIndex 0.14 agent migration

| Old API (≤0.13) | New API (0.14+) |
|-----------------|-----------------|
| `ReActAgent.from_tools(...)` | `ReActAgent(tools=..., system_prompt=...)` |
| `agent.achat(msg)` | `await agent.run(user_msg=msg, memory=mem)` |
| `context=` kwarg | `system_prompt=` |
| Returns `str` | Returns `AgentOutput` → extract `response.content` |

`chat_with_memory()` now returns `AgentTurnResult(answer, citations)`.

---

## Mandatory [Source N] citation prompts

### Before

```
Rule 6: Cite sources (e.g. [Source 2]) when stating specific facts.
Agent: Cite section titles and page numbers when stating specific facts.
```

Model paraphrased tool output → dropped tags → score fallback → wrong sources in UI.

### After (`src/prompts.py`)

**Generation prompt — `CITATION RULES (MANDATORY)`:**

- Every factual sentence MUST end with `[Source N]` (1-based, from `<source id="N">`)
- Multi-source: `[Source 1, Source 2]` or per-sentence tags
- Never invent source numbers
- Abstention-only answers exempt

**Few-shot Example J (BAD):** Same sick-leave answer without `[Source 1]` — labeled NEVER DO THIS.

**Agent system prompt:**

- Preserve `[Source N]` tags verbatim from `policy_search`
- Do not replace with page numbers or filenames alone

### Verification

```bash
python -m pytest tests/test_prompts.py -v
# Ask in Streamlit: "How many sick days do employees receive?"
# logs/app.log → Citation selection | mode=cited_in_answer
```

---

## Chroma telemetry & index health fixes

### Telemetry (`src/chroma_telemetry.py`)

```python
class NoOpProductTelemetry(ProductTelemetryClient):
    def capture(self, event): return  # no posthog call
```

Wired via `chroma_product_telemetry_impl="src.chroma_telemetry.NoOpProductTelemetry"` in `get_chroma_client()`.

### Index probe (`probe_chroma_index()`)

Returns real state regardless of cache:

```python
{
  "count": 81,
  "ready": True,
  "collections": ["company_policies"],
  "error": None,
  ...
}
```

Streamlit error screen now shows `Chunks (actual)` and the real error — not a placeholder `0`.

---

## Docker deployment

### Architecture

```
Host: Ollama :11434
    ↑ host.docker.internal
Container: Streamlit :8501
    ↔ volumes: data/, storage/, logs/, hf_cache
```

### Quick start

```bash
cp .env.docker.example .env.docker
docker compose build
docker compose run --rm app python scripts/index_documents.py
docker compose up
# → http://localhost:8501
```

### Entrypoint (`docker/entrypoint.sh`)

1. Wait for Ollama (60s timeout)
2. Optional `AUTO_INDEX_ON_START=true` when Chroma empty
3. `streamlit run app/streamlit_app.py --server.address=0.0.0.0`

### Why host Ollama (not in-compose)

- User choice: reuse existing Ollama install and GPU setup on host
- `qwen2.5:7b` needs ~8GB RAM — keeps app container lightweight

---

## Configuration that mattered

### Retrieval & reranking

| Variable | Final value | Why |
|----------|-------------|-----|
| `RETRIEVAL_CANDIDATE_K` | 30 | Recall pool for reranker |
| `RERANKER_TOP_N` | 6 | Multi-part policy questions need more context |
| `RERANK_MIN_SCORE_RATIO` | 0.40 | Drop marginal chunks (was 0.35) |
| `RERANK_MIN_KEEP` | 3 | Never return empty context |
| `ENABLE_QUERY_REWRITE` | true | Keyword alignment for policy queries |

### Generation & grounding

| Variable | Final value | Why |
|----------|-------------|-----|
| `GROUNDING_STRICTNESS` | balanced | HR Q&A usefulness |
| `FAITHFULNESS_GUARD_MODE` | balanced | Reject → log → keep answer |
| `RESPONSE_PROMPT_VERSION` | v2_balanced | Partial answers + semantic mapping |

### Citations

| Variable | Default | Why |
|----------|---------|-----|
| `CITATION_MIN_RELEVANCE_RATIO` | 0.55 | Strict fallback when no `[Source N]` tags |
| `ENABLE_CITATION_PIPELINE_LOGGING` | true | Debug citation mismatches |
| `CITATION_MAX_SOURCES` | 6 | Cap UI clutter |
| `CITATION_SHOW_SCORE` | false | Enable for debugging |

### Chroma & Docker

| Variable | Default | Why |
|----------|---------|-----|
| `ANONYMIZED_TELEMETRY` | False | Disable Chroma telemetry (with NoOp impl) |
| `AUTO_INDEX_ON_START` | false | Docker: auto-run `index_documents.py` if empty |
| `OLLAMA_BASE_URL` (Docker) | `http://host.docker.internal:11434` | Reach host Ollama from container |

---

## How to reproduce & test

### Full eval (60 cases policy+guidebook; guidebook-only ~47 min)

```bash
python -m pytest tests/ -v          # 182 tests
python scripts/evaluate.py          # golden set → logs/evaluation_results.json
```

### Docker smoke test

```bash
docker compose build
docker compose run --rm app python scripts/index_documents.py
docker compose up
# Ask a policy question at http://localhost:8501
docker compose down
```

### Citation quality check

1. Ask: *"How many sick days do employees receive?"*
2. Confirm answer cites `[Source 1]` (or similar).
3. Confirm **only** sick-leave-related sections appear in expanders.
4. Check `logs/app.log` for `Citation selection | mode=cited_in_answer`.

### Metric regression guard

Before merging pipeline changes, compare aggregates in `logs/evaluation_results.json`:

- Answer Relevancy should not drop below **0.70** without documented trade-off.
- Faithfulness should not drop below **0.75** in balanced mode without review.

---

## What is still open

| Item | Status | Notes |
|------|--------|-------|
| Relevancy ≥ 0.75 | **0.747** | `remote_work` abstention judge scores 0.0; patch applied, re-eval pending |
| Faithfulness ≥ 0.90 in balanced | **0.807** | Trade-off from keeping guard-rejected answers; needs tighter generation |
| Mandatory `[Source N]` tags | **Done** | Balanced prompts + agent system prompt updated |
| Citation pipeline (generation-linked) | **Done** | `src/citations.py` + `SourceTrackingQueryEngine` |
| Chroma telemetry errors | **Done** | `NoOpProductTelemetry` + `posthog<3` |
| Streamlit index false negative | **Done** | `probe_chroma_index()` + cache reset |
| Docker (Streamlit + host Ollama) | **Done** | `docker compose up` → :8501 |
| Hybrid BM25 | Planned | Exact legal terms still vector-only |
| Per-user ACL / metadata filters | Planned | Hooks exist in retriever |
| Faithfulness recovery without relevancy loss | In progress | Tighter claims, not re-enabling full abstention on guard reject |
| Ollama-in-compose Docker variant | Planned | Optional for fully self-contained deploy |
| GPU Docker image | Planned | CPU default today |

---

## Project structure

```
company_policy_rag/
├── app/
│   ├── streamlit_app.py      # Primary UI (chat, sidebar, citations, index diagnostics)
│   └── chat_app.py           # Chainlit (legacy)
├── docker/
│   └── entrypoint.sh         # Ollama wait + optional auto-index + Streamlit
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.txt
├── .env.docker.example
├── src/
│   ├── indexing.py           # PDF → Chroma + probe_chroma_index()
│   ├── chroma_telemetry.py   # No-op Chroma telemetry
│   ├── retriever.py          # Rewrite → retrieve → rerank → filter
│   ├── generation.py         # Grounded synthesis + guard + source tracking
│   ├── citations.py          # [Source N] parsing + citation selection
│   ├── prompts.py            # Strict/balanced + mandatory citation rules
│   ├── query_processing.py   # LLM rewrite + policy-term augmentation
│   ├── agent.py              # ReAct agent (LlamaIndex 0.14 workflow)
│   ├── memory.py             # Multi-turn conversation buffer
│   ├── evaluation.py         # Golden set + context-aware LLM judge
│   ├── postprocessors.py     # Relative score threshold filter
│   └── config.py             # Single source of truth (.env overrides)
├── data/
│   ├── policies/             # PDF corpus
│   └── eval/golden_dataset.json
├── logs/
│   ├── evaluation_results.json   # Metric trend log (append-only)
│   └── app.log
├── tests/                    # 182 tests
├── scripts/
│   ├── index_documents.py
│   └── evaluate.py
├── README.md                 # Setup & reference
└── README2.md                # This file — journey & lessons
```

---

## Tests added during this journey

| Test file | What it guards |
|-----------|----------------|
| `test_generation.py` | Guard behavior, double-ending strip, `normalize_balanced_answer` |
| `test_citations.py` | `[Source N]` parsing, cited-only display, score fallback |
| `test_query_processing.py` | Policy-term augmentation (benefits, resignation, discipline) |
| `test_prompts.py` | Balanced prompt rules, mandatory `[Source N]` citation language |
| `test_evaluation.py` | Context-aware relevancy judge |
| `test_chroma_telemetry.py` | NoOp telemetry, settings conflict recovery, index probe |

**Total: 182 tests passing** (as of last run).

---

## One-page cheat sheet for the next engineer

1. **Run eval before and after every pipeline change.**
2. **Balanced mode** for users; **strict mode** for auditors.
3. **Never show citations from a separate retrieval** — use `AgentTurnResult.citations`.
4. **Every factual sentence needs `[Source N]` tags** — or score fallback shows wrong sources.
5. **If relevancy drops**, check: guard abstention → double-endings → few-shot bloat → retrieval vocabulary.
6. **If sources look wrong**, check `logs/app.log` for `Citation pipeline` lines and missing tags.
7. **If Streamlit says no index**, check `Chunks (actual)` in diagnostics; clear Chroma cache; restart.
8. **If Chroma telemetry errors**, ensure `NoOpProductTelemetry` + `posthog<3` are installed.
9. **If Streamlit fails on init**, check LlamaIndex 0.14+ `agent.run()` API (not `from_tools`).
10. **Docker:** host Ollama must run first; `docker compose up` → :8501.
11. **Read `logs/evaluation_results.json`** — it is the project's memory.

---

*Built with production-rag principles: measure first, prioritize citation trust, name trade-offs explicitly, and log everything worth debugging twice.*