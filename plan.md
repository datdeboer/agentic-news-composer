# Agentic News Composer — Plan

## Project Goal

Build a daily news collector and aggregator that automatically fetches articles from multiple sources, summarizes the top stories, compiles relevant links, and generates three blog drafts in different styles. A human reviews and approves each draft through a Streamlit UI before the output is finalized.

The primary learning objective is hands-on experience with LangChain and LangGraph patterns: graph state, nodes, parallel fan-out, the Send API, interrupt/resume, and checkpointing.

---

## Functional Requirements

### News Collection

- Fetch articles from RSS/Atom feeds defined in a YAML config file
- Scrape article content from sites that lack RSS feeds using HTML parsing
- Pull social signals (HN top stories, Reddit hot posts) from public APIs — no API keys required
- All sources fetched in parallel on each run
- Deduplicate articles by URL before processing

### Article Ranking

- Rank fetched articles by relevance to user-defined topics (configured in YAML)
- Topics are keywords such as "artificial intelligence", "climate technology", etc.
- Ranking is performed by an LLM, producing a 0–10 relevance score per article
- Only the top-ranked articles proceed to summarization and link selection

### Digest Generation

- Produce exactly 5 article summaries, each 2–3 sentences
- Produce exactly 5 trending links, each with a one-sentence reason for inclusion
- Both are generated in parallel after ranking

### Blog Draft Generation

- Generate 3 blog drafts in distinct styles from the same news digest:
  1. **Opinion/Editorial** — opinionated, first-person, takes a clear stance
  2. **Newsletter Recap** — warm, conversational, highlights the top 5 stories
  3. **Deep Dive** — analytical, thorough, explores one story in depth with context and implications
- All 3 drafts are generated in parallel

### Human Review Workflow

- Pipeline pauses after draft generation and presents all 3 drafts to the user
- User can independently approve or request changes on each draft
- Revision requests include free-text notes explaining the desired changes
- Only flagged drafts are regenerated; approved drafts are preserved
- The review cycle repeats until all 3 drafts are approved
- Once all approved, the finalized digest and drafts are saved to disk

### Output

- Save finalized output as a Markdown file at `output/<YYYY-MM-DD>.md`
- File contains: top 5 summaries with links, top 5 trending links, and all 3 approved blog drafts

### User Interface

- Streamlit web UI with three sections:
  1. **Controls** — display configured topics, trigger a run, show progress
  2. **Digest view** — top 5 summaries and trending links
  3. **Blog draft review** — one tab per draft style, each with approve and request-changes actions
- Progress indicators while the pipeline is running
- UI remains responsive during background processing

### Configuration

- RSS feed list is configurable via `config/feeds.yaml`
- Topic keywords are configurable via `config/topics.yaml`
- Reddit subreddits and HN story count are also configurable in `config/topics.yaml`
- LLM model is configurable via environment variable (`OPENROUTER_MODEL`)
- Runs are triggered manually by the user via the UI

---

## Technical Requirements

### LLM Integration

- All LLM calls use `ChatOpenAI` from `langchain-openai` pointing to OpenRouter
- Base URL: `https://openrouter.ai/api/v1`
- Authentication via `OPENROUTER_API_KEY` environment variable
- Default model: `openai/gpt-4o-mini` (overridable via `OPENROUTER_MODEL`)
- LLM used for: article ranking, summarization, link selection, blog draft generation

### LangGraph Architecture

The pipeline is implemented as a `StateGraph` with the following node sequence:

```
fetch_sources
    → rank_and_filter
    → summarize  \
                  → join_digest
    → compile_links /
    → [Send API fan-out] → write_draft (x3 in parallel)
    → collect_drafts
    → human_review          ← interrupt()
    → [conditional]
        all approved  → finalize → END
        some revisions → [Send API fan-out] → rewrite_draft (flagged only)
                          → collect_drafts → human_review   ← loops
```

**LangGraph patterns exercised:**

| Pattern | Where used |
|---------|-----------|
| `StateGraph` + `TypedDict` state | entire graph |
| Parallel edges | `rank_and_filter` → `summarize` + `compile_links` |
| `Send` API (map-reduce fan-out) | `join_digest` → 3x `write_draft` |
| `interrupt(value)` | `human_review` node — pauses graph, surfaces data to UI |
| `Command(resume=...)` | Streamlit resumes graph with feedback list |
| Conditional edge | `_route_after_review` — "finalize" or `list[Send]` to `rewrite_draft` |
| Cycle (loop) | `rewrite_draft` → `collect_drafts` → `human_review` repeats until all approved |
| `SqliteSaver` checkpointer | persists graph state across Streamlit rerenders |
| `graph.stream()` | real-time progress events in UI |

### State Schema

```python
class NewsComposerState(TypedDict):
    topics: list[str]
    raw_articles: list[dict]       # {title, url, content, source, score}
    top_5_summaries: list[dict]    # {title, url, summary}
    top_5_links: list[dict]        # {title, url, reason}
    blog_drafts: list[dict]        # [{style, title, content}, x3]
    human_feedback: dict | None    # [{style, action: approve|revise, notes}]
    finalized: bool
```

### News Sources

| Source | Implementation | Notes |
|--------|----------------|-------|
| RSS/Atom feeds | `feedparser` | URLs in `config/feeds.yaml`, 10 articles per feed |
| Web scraping | `requests` + `beautifulsoup4` | For sites without RSS |
| Hacker News | HN Algolia REST API | No key required, configurable count |
| Reddit | Reddit JSON API (`/r/<sub>/hot.json`) | No key required, configurable subreddits |

All sources fetched via `asyncio.gather` inside the `fetch_sources` node.

### Streamlit State Management

- `st.session_state` holds `thread_id`, graph reference, run status, and interrupt data
- Graph runs in a background `threading.Thread` to avoid blocking the UI
- Streamlit polls graph state via `graph.get_state(config)` and calls `st.rerun()` to refresh
- On resume, Streamlit calls `graph.stream(Command(resume=feedback), config=...)` in a thread

### Checkpointing

- `SqliteSaver` stores graph state in `output/checkpoints.db`
- Enables state recovery if the Streamlit process restarts mid-run
- Each run uses a unique `thread_id` (UUID) for isolation

### Key Dependencies

```
langchain>=0.3
langchain-openai>=0.2
langgraph>=0.2
langgraph-checkpoint-sqlite>=2.0
streamlit>=1.35
feedparser>=6.0
beautifulsoup4>=4.12
requests>=2.31
pyyaml>=6.0
python-dotenv>=1.0
aiohttp>=3.9
```

---

## Implementation Milestones

### M1 — Project Scaffold + Raw Fetch

Sources, config files, and `NewsComposerState` defined. `fetch_sources` node fetches and deduplicates articles from all three source types.

Verify: run fetch_sources, print count of unique articles.

### M2 — LLM Pipeline: Rank, Summarize, Links

LLM client wired to OpenRouter. `rank_and_filter`, `summarize`, and `compile_links` nodes implemented. Partial graph runs fetch → rank → parallel summarize/links.

Verify: graph produces 5 summaries and 5 links in state.

### M3 — Blog Draft Fan-Out

`write_draft` node with Send API fan-out for 3 styles. `collect_drafts` join node. `finalize` writes output file. Full pipeline runs end-to-end with `MemorySaver`.

Verify: `python run.py` writes `output/<date>.md` with all 3 drafts.

### M4 — Human-in-the-Loop

`human_review` node uses `interrupt()`. `_route_after_review` conditional edge routes to finalize or `rewrite_draft` fan-out. `rewrite_draft` incorporates revision notes. Switch to `SqliteSaver`.

Verify: pipeline pauses, partial approval triggers targeted regeneration, full approval writes output.

### M5 — Streamlit UI

Three-section layout: controls, digest view, draft review tabs. Background thread runs graph. Polling detects interrupt. Submit resumes graph with feedback. Progress shown during processing.

Verify: `streamlit run app.py` — full end-to-end flow through the UI.
