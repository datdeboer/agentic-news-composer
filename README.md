# Agentic News Composer

A production-style **agentic AI pipeline** that fetches daily tech news from multiple sources, ranks and summarizes articles with an LLM, generates three parallel blog drafts in distinct writing styles, and routes them through a human-in-the-loop review workflow — all orchestrated with **LangGraph** and surfaced through a **React + FastAPI** UI.

---

## What it does

1. **Fetches** articles in parallel from RSS feeds, scraped websites, Hacker News, and Reddit
2. **Ranks** them by relevance to user-defined topic keywords using an LLM
3. **Summarizes** the top 5 stories and compiles 5 trending links — in parallel
4. **Generates** 3 blog drafts simultaneously (Opinion, Newsletter Recap, Deep Dive) via LangGraph's Send API
5. **Pauses** for human review: approve drafts or request targeted revisions with free-text notes
6. **Loops** — only flagged drafts are regenerated; approved ones are preserved
7. **Finalizes** the full digest to a dated Markdown file once all drafts are approved
8. **Emails** the newsletter via the Brevo API (optional)

---

## Technical highlights

| Area | Implementation |
|---|---|
| **Agentic orchestration** | LangGraph `StateGraph` with typed state, parallel edges, and a revision cycle |
| **Parallel fan-out** | LangGraph `Send` API dispatches 3 simultaneous draft-writing nodes (map-reduce) |
| **Human-in-the-loop** | `interrupt()` pauses the graph; `Command(resume=...)` resumes with structured feedback |
| **Checkpointing** | `SqliteSaver` persists graph state across requests; each run isolated by UUID thread ID |
| **Conditional routing** | `_route_after_review` returns `"finalize"` or a dynamic `list[Send]` based on feedback |
| **Async data fetching** | `asyncio.gather` fetches all sources concurrently inside a LangGraph node |
| **LLM integration** | Centralized `graph/llm.py` factory — switch between OpenRouter and Groq at runtime |
| **Streaming UI** | FastAPI streams `graph.stream()` events as SSE; React consumes them to update the UI live |
| **Pipeline visibility** | Sidebar shows each pipeline step lighting up in real time as nodes complete |
| **Email newsletter** | Digest emailed via Brevo REST API after each approved run (optional) |

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

## UI

The frontend is a **React + Tailwind CSS** single-page app served by the **FastAPI** backend.

### Pipeline sidebar

A sidebar on the left tracks every step of the graph in real time. Each step card turns green as the corresponding LangGraph node completes, and turns amber when the pipeline is paused waiting for your review. This gives immediate visual feedback on where the run is in the pipeline without needing to read log output.

### Human-in-the-loop review

When all three drafts are ready, the pipeline pauses and presents a tabbed review panel. Each draft can be independently approved or sent back for revision with free-text notes. Only flagged drafts are regenerated; approved ones are kept.

### LLM provider selector

Before starting a run, choose between **OpenRouter** and **Groq** as the LLM provider, and pick a specific model from each provider's list. The selection is applied to the entire run.

---

## Stack

**Backend**
- **Python 3.11+**
- **FastAPI + uvicorn** — REST API and SSE streaming
- **LangGraph** — agentic graph orchestration, interrupt/resume, checkpointing
- **LangChain / ChatOpenAI** — LLM calls via [OpenRouter](https://openrouter.ai) or [Groq](https://console.groq.com)
- **feedparser** — RSS/Atom feed ingestion
- **BeautifulSoup4 + requests** — HTML scraping and Brevo API calls
- **aiohttp / asyncio** — concurrent source fetching
- **SQLite** — persistent graph checkpoints (`SqliteSaver`)

**Frontend**
- **React 18 + Vite** — component-based UI with fast dev builds
- **Tailwind CSS** — utility-first styling
- **react-markdown + remark-gfm** — renders blog drafts from markdown

---

## Quickstart

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY and/or GROQ_API_KEY
# Optionally set BREVO_* variables to enable email delivery

# 3. Install and build the frontend
cd frontend
npm install
npm run build
cd ..

# 4a. Start the API server (serves built React UI at http://localhost:8000)
uvicorn api:app --reload

# 4b. Or run headless (auto-approves all drafts, no UI needed)
python run.py

# 4c. Or run with interactive CLI review
python run.py --interactive
```

> **Frontend dev mode:** run `npm run dev` inside `frontend/` for hot-reload on port 5173 while the API runs on port 8000.

Output is saved to `output/<YYYY-MM-DD>-digest.md`.

---

## Configuration

| File | Purpose |
|---|---|
| `config/feeds.yaml` | RSS/Atom feed URLs to ingest |
| `config/topics.yaml` | Topic keywords for relevance ranking, subreddits, HN story count |
| `.env` | API keys and email settings (see `.env.example`) |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | One of these | LLM calls via OpenRouter |
| `GROQ_API_KEY` | One of these | LLM calls via Groq |
| `BREVO_API_KEY` | Optional | Enables email delivery after each run |
| `BREVO_FROM_EMAIL` | Optional | Verified sender address in Brevo |
| `BREVO_FROM_NAME` | Optional | Sender display name |
| `BREVO_TO_EMAILS` | Optional | Comma-separated recipient addresses |

---

## Project structure

```
agentic-news-composer/
├── api.py                     # FastAPI backend — SSE streaming + REST endpoints
├── run.py                     # Headless CLI runner
├── graph/
│   ├── graph.py               # LangGraph definition + compilation
│   ├── llm.py                 # Centralized LLM factory (provider/model switching)
│   ├── state.py               # NewsComposerState TypedDict
│   └── nodes/
│       ├── fetch_sources.py   # Parallel fetch (RSS + scrape + HN/Reddit)
│       ├── rank_and_filter.py # LLM relevance ranking
│       ├── summarize.py       # LLM summarization (top 5)
│       ├── compile_links.py   # LLM link curation (top 5)
│       ├── draft_blog_posts.py# Send API fan-out → 3 parallel draft nodes
│       ├── human_review.py    # interrupt() + structured feedback collection
│       ├── regenerate_drafts.py # Rewrites only flagged drafts
│       ├── finalize.py        # Writes output/<date>.md
│       └── send_email.py      # Brevo API email delivery
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Root component — status state machine, SSE consumer
│   │   ├── api.js             # Fetch helpers + SSE async generator parser
│   │   └── components/
│   │       ├── Pipeline.jsx   # Pipeline sidebar — steps light up as nodes complete
│   │       ├── ProviderSelector.jsx # LLM provider + model dropdowns
│   │       ├── ReviewDrafts.jsx    # Tabbed HITL review panel
│   │       └── Digest.jsx         # Top 5 summaries + trending links
│   └── package.json
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
| `Command(resume=...)` | FastAPI `/review` endpoint resumes graph with structured feedback from React UI |
| Conditional edge returning `list[Send]` | Routes to targeted `rewrite_draft` nodes per flagged style |
| Revision cycle (loop) | `rewrite_draft → collect_drafts → human_review` repeats until fully approved |
| `SqliteSaver` checkpointer | State survives process restarts; each run isolated by UUID thread ID |
| `graph.stream()` | Streams node-completion events as SSE to the React UI in real time |
