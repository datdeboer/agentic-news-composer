# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`agentic-news-composer` is an early-stage project. This CLAUDE.md should be updated as the codebase grows.

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env

# Run headless CLI (auto-approves all drafts)
python run.py

# Run CLI with interactive review
python run.py --interactive

# Start FastAPI backend (port 8000)
uvicorn api:app --reload

# Install frontend dependencies (first time only)
cd frontend && npm install

# Start React dev server (port 5173) — proxies /api to :8000
cd frontend && npm run dev

# Build React app for production
cd frontend && npm run build
```

## Architecture

```
agentic-news-composer/
├── api.py                        # FastAPI backend (SSE streaming, graph runner)
├── app.py                        # Legacy Streamlit UI (kept for reference)
├── run.py                        # Headless CLI runner
├── frontend/                     # React + Vite + Tailwind frontend
│   ├── src/
│   │   ├── App.jsx               # Root component + state machine
│   │   ├── api.js                # Fetch + SSE helpers
│   │   └── components/
│   │       ├── Pipeline.jsx      # Pipeline steps sidebar
│   │       ├── Digest.jsx        # Summaries + trending links
│   │       └── ReviewDrafts.jsx  # Human-in-the-loop review UI
│   └── package.json
├── graph/
│   ├── graph.py                  # LangGraph graph definition + compilation
│   ├── state.py                  # NewsComposerState TypedDict
│   └── nodes/
│       ├── fetch_sources.py      # Parallel RSS + scrape + HN/Reddit fetch
│       ├── rank_and_filter.py    # LLM ranks articles by topic relevance
│       ├── summarize.py          # LLM generates 5 summaries
│       ├── compile_links.py      # Selects 5 trending/popular links
│       ├── draft_blog_posts.py   # Fan-out: 3 parallel blog draft nodes (Send API)
│       ├── human_review.py       # interrupt/resume + conditional routing
│       ├── regenerate_drafts.py  # Regenerates flagged drafts only
│       └── finalize.py           # Writes output/<date>.md
├── sources/
│   ├── rss_fetcher.py            # feedparser-based RSS/Atom reader
│   ├── web_scraper.py            # requests + BeautifulSoup scraper
│   └── social_fetcher.py        # HN Algolia API + Reddit JSON API
├── config/
│   ├── feeds.yaml                # List of RSS feed URLs
│   └── topics.yaml               # User-defined topic keywords
└── output/                       # Saved digests (gitignored)
```

### LLM

All LLM calls use `ChatOpenAI` pointed at OpenRouter (`https://openrouter.ai/api/v1`).
Set `OPENROUTER_API_KEY` and optionally `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`).

### LangGraph patterns used

- `StateGraph` with `TypedDict` state
- Parallel edges (summarize + compile_links fan-out after rank_and_filter)
- `Send` API for dynamic fan-out to 3 draft nodes (map-reduce)
- `interrupt(value)` — pauses graph, surfaces drafts to React UI via SSE
- `Command(resume=feedback)` — resumes with human feedback
- Conditional routing via `should_finalize`
- Revision cycle (loop back to human_review until all drafts approved)
- `SqliteSaver` checkpointer at `output/checkpoints.db`
