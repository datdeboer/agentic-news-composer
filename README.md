# Agentic News Composer

A production-style **agentic AI pipeline** that fetches daily tech news from multiple sources, ranks and summarizes articles with an LLM, generates three parallel blog drafts in distinct writing styles, and routes them through a human-in-the-loop review workflow — all orchestrated with **LangGraph** and surfaced through a **Streamlit** UI.

---

## What it does

1. **Fetches** articles in parallel from RSS feeds, scraped websites, Hacker News, and Reddit
2. **Ranks** them by relevance to user-defined topic keywords using an LLM
3. **Summarizes** the top 5 stories and compiles 5 trending links — in parallel
4. **Generates** 3 blog drafts simultaneously (Opinion, Newsletter Recap, Deep Dive) via LangGraph's Send API
5. **Pauses** for human review: approve drafts or request targeted revisions with free-text notes
6. **Loops** — only flagged drafts are regenerated; approved ones are preserved
7. **Finalizes** the full digest to a dated Markdown file once all drafts are approved

---

## Technical highlights

| Area | Implementation |
|---|---|
| **Agentic orchestration** | LangGraph `StateGraph` with typed state, parallel edges, and a revision cycle |
| **Parallel fan-out** | LangGraph `Send` API dispatches 3 simultaneous draft-writing nodes (map-reduce) |
| **Human-in-the-loop** | `interrupt()` pauses the graph; `Command(resume=...)` resumes with structured feedback |
| **Checkpointing** | `SqliteSaver` persists graph state — survives Streamlit process restarts mid-run |
| **Conditional routing** | `_route_after_review` returns `"finalize"` or a dynamic `list[Send]` based on feedback |
| **Async data fetching** | `asyncio.gather` fetches all sources concurrently inside a LangGraph node |
| **LLM integration** | `ChatOpenAI` via OpenRouter — model and API key configurable via environment variables |
| **UI** | Streamlit with session recovery, live progress via `st.status()`, and tabbed draft review |

---

## Architecture

```
                      ┌─────────────────┐
                      │  fetch_sources  │  RSS + scrape + HN + Reddit (async)
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │ rank_and_filter │  LLM scores articles 0–10 by topic relevance
                      └──────┬──────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
      ┌────────▼────────┐        ┌──────────▼──────────┐
      │    summarize    │        │    compile_links     │  (parallel)
      └────────┬────────┘        └──────────┬──────────┘
               │                            │
               └─────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   join_digest   │
                    └────────┬────────┘
                             │  Send API fan-out
             ┌───────────────┼───────────────┐
             │               │               │
     ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼────────┐
     │ write_draft  │ │ write_draft │ │ write_draft  │  (parallel)
     │  [opinion]   │ │[newsletter] │ │ [deep-dive]  │
     └───────┬──────┘ └──────┬──────┘ └─────┬────────┘
             └───────────────┼───────────────┘
                             │
                    ┌────────▼────────┐
                    │ collect_drafts  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  human_review   │  ← interrupt() — UI presents drafts
                    └────────┬────────┘
                             │  conditional routing
              ┌──────────────┴──────────────┐
              │ all approved                │ some flagged
     ┌────────▼────────┐           ┌────────▼────────┐
     │    finalize     │           │  rewrite_draft  │  (Send fan-out, flagged only)
     └────────┬────────┘           └────────┬────────┘
              │                             │
             END                    collect_drafts → human_review  (loops)
```

---

## Stack

- **Python 3.11+**
- **LangGraph** — agentic graph orchestration, interrupt/resume, checkpointing
- **LangChain / ChatOpenAI** — LLM calls via [OpenRouter](https://openrouter.ai)
- **Streamlit** — interactive UI with live progress and session recovery
- **feedparser** — RSS/Atom feed ingestion
- **BeautifulSoup4 + requests** — HTML scraping
- **aiohttp / asyncio** — concurrent source fetching
- **SQLite** — persistent graph checkpoints (`SqliteSaver`)

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL)

# 3a. Run the Streamlit UI
streamlit run app.py

# 3b. Or run headless (auto-approves all drafts)
python run.py

# 3c. Or run with interactive CLI review
python run.py --interactive
```

Output is saved to `output/<YYYY-MM-DD>-digest.md`.

---

## Configuration

| File | Purpose |
|---|---|
| `config/feeds.yaml` | RSS/Atom feed URLs to ingest |
| `config/topics.yaml` | Topic keywords for relevance ranking, subreddits, HN story count |
| `.env` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`) |

---

## Project structure

```
agentic-news-composer/
├── app.py                     # Streamlit UI
├── run.py                     # Headless CLI runner
├── graph/
│   ├── graph.py               # LangGraph definition + compilation
│   ├── state.py               # NewsComposerState TypedDict
│   └── nodes/
│       ├── fetch_sources.py   # Parallel fetch (RSS + scrape + HN/Reddit)
│       ├── rank_and_filter.py # LLM relevance ranking
│       ├── summarize.py       # LLM summarization (top 5)
│       ├── compile_links.py   # LLM link curation (top 5)
│       ├── draft_blog_posts.py# Send API fan-out → 3 parallel draft nodes
│       ├── human_review.py    # interrupt() + structured feedback collection
│       ├── regenerate_drafts.py # Rewrites only flagged drafts
│       └── finalize.py        # Writes output/<date>.md
├── sources/
│   ├── rss_fetcher.py         # feedparser RSS reader
│   ├── web_scraper.py         # requests + BeautifulSoup scraper
│   └── social_fetcher.py      # HN Algolia API + Reddit JSON API
└── config/
    ├── feeds.yaml
    └── topics.yaml
```

---

## LangGraph patterns demonstrated

| Pattern | Where |
|---|---|
| `StateGraph` + `TypedDict` | Entire pipeline |
| Parallel edges | `rank_and_filter` → `summarize` + `compile_links` simultaneously |
| `Send` API (map-reduce) | Fan-out to 3 independent `write_draft` nodes |
| `interrupt(value)` | Pauses graph at `human_review`, surfaces draft data to UI |
| `Command(resume=...)` | Streamlit resumes graph with structured feedback |
| Conditional edge returning `list[Send]` | Routes to targeted `rewrite_draft` nodes per flagged style |
| Revision cycle (loop) | `rewrite_draft → collect_drafts → human_review` repeats until fully approved |
| `SqliteSaver` checkpointer | State survives process restarts; each run isolated by UUID thread ID |
| `graph.stream()` | Streams node-completion events to the UI in real time |
