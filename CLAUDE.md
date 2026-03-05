# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`agentic-news-composer` is an early-stage project. This CLAUDE.md should be updated as the codebase grows.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.example .env

# Run headless CLI (auto-approves all drafts)
python run.py

# Run CLI with interactive review
python run.py --interactive

# Launch Streamlit UI
streamlit run app.py
```

## Architecture

```
agentic-news-composer/
├── app.py                        # Streamlit UI entry point
├── run.py                        # Headless CLI runner
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
- `interrupt(value)` — pauses graph, surfaces drafts to Streamlit
- `Command(resume=feedback)` — resumes with human feedback
- Conditional routing via `should_finalize`
- Revision cycle (loop back to human_review until all drafts approved)
- `SqliteSaver` checkpointer at `output/checkpoints.db`
