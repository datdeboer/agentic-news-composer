"""LangGraph node: write finalized digest + approved drafts to output file."""
from datetime import date
from pathlib import Path

from output.paths import dated_path
from graph.state import NewsComposerState

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def finalize(state: NewsComposerState) -> dict:
    """Write summaries, links, and blog drafts to dated output files."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    summaries = state.get("top_5_summaries", [])
    links = state.get("top_5_links", [])
    drafts = state.get("blog_drafts", [])

    _save_summaries(summaries, links)
    _save_digest(summaries, links, drafts)

    return {"finalized": True}


def _save_summaries(summaries: list[dict], links: list[dict]) -> None:
    path = dated_path("summaries")
    lines = ["# Summaries & Trending Links\n"]

    lines.append("## Top 5 Summaries\n")
    for i, s in enumerate(summaries, 1):
        lines.append(f"### {i}. [{s.get('title', '')}]({s.get('url', '')})")
        lines.append(f"{s.get('summary', '')}\n")

    lines.append("## Trending Links\n")
    for i, l in enumerate(links, 1):
        lines.append(f"{i}. **[{l.get('title', '')}]({l.get('url', '')})**")
        lines.append(f"   {l.get('reason', '')}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[finalize] Summaries saved to {path}", flush=True)


def _save_digest(summaries: list[dict], links: list[dict], drafts: list[dict]) -> None:
    path = dated_path("digest")
    today = date.today().isoformat()
    lines = [f"# News Digest — {today}\n"]

    lines.append("## Top 5 Stories\n")
    for i, s in enumerate(summaries, 1):
        lines.append(f"### {i}. [{s.get('title', '')}]({s.get('url', '')})\n")
        lines.append(f"{s.get('summary', '')}\n")

    lines.append("## Trending Links\n")
    for i, l in enumerate(links, 1):
        lines.append(f"{i}. **[{l.get('title', '')}]({l.get('url', '')})**  — {l.get('reason', '')}\n")

    for draft in drafts:
        style = draft.get("style", "unknown")
        lines.append(f"\n---\n\n## Blog Draft: {style.title()}\n")
        lines.append(draft.get("content", "") + "\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[finalize] Digest saved to {path}", flush=True)
