"""LangGraph nodes: generate blog drafts using Send API fan-out (3 parallel styles)."""
import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.types import Send

from graph.state import NewsComposerState

logger = logging.getLogger(__name__)

BLOG_STYLES = ["opinion", "newsletter", "deep-dive"]

STYLE_PROMPTS = {
    "opinion": """Write a compelling opinion/editorial piece (600-900 words) about today's top tech news.
Take a clear stance, be provocative and thought-provoking. Use first person voice.
Structure: hook opening → thesis → 2-3 argument paragraphs → counter-argument → strong conclusion.
Return ONLY the blog post in markdown format, starting with a # title.""",

    "newsletter": """Write a friendly, engaging newsletter recap (500-700 words) covering today's top tech stories.
Use a warm, conversational tone — like writing to a friend who's into tech.
Structure: brief intro → 5 story highlights with context → 5 links section → closing thought.
Return ONLY the newsletter in markdown format, starting with a # title.""",

    "deep-dive": """Write a thorough deep-dive analysis (800-1200 words) on the most significant tech story today.
Be analytical, include context, implications, and expert perspective.
Structure: executive summary → background → main analysis → implications → what to watch next.
Return ONLY the article in markdown format, starting with a # title.""",
}


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )


def _build_context(state: dict) -> str:
    """Build news context string from summaries and links."""
    summaries = state.get("top_5_summaries", [])
    links = state.get("top_5_links", [])

    context = "## Today's Top Stories\n\n"
    for i, s in enumerate(summaries, 1):
        context += f"**{i}. {s['title']}**\n{s['summary']}\nURL: {s['url']}\n\n"

    context += "## Trending Links\n\n"
    for i, l in enumerate(links, 1):
        context += f"**{i}. {l['title']}** — {l['reason']}\nURL: {l['url']}\n\n"

    return context


def fan_out_drafts(state: NewsComposerState) -> list[Send]:
    """Fan-out: send one draft task per blog style using the Send API."""
    print("[fan_out_drafts] Fanning out to 3 parallel draft nodes...")
    return [
        Send("write_draft", {**state, "draft_style": style})
        for style in BLOG_STYLES
    ]


def write_draft(state: dict) -> dict:
    """Write a single blog draft for the given style."""
    style = state["draft_style"]
    print(f"[write_draft:{style}] Generating {style} draft...")

    context = _build_context(state)
    style_prompt = STYLE_PROMPTS[style]

    prompt = f"""{style_prompt}

Here is today's news digest to draw from:

{context}"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Extract title from first markdown heading
    lines = content.split("\n")
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("#")), f"{style.title()} Draft")

    draft = {
        "style": style,
        "title": title,
        "content": content,
    }

    print(f"[write_draft:{style}] Done — '{title}'", flush=True)
    # Return only this draft — the _merge_drafts reducer in state.py handles merging
    # across parallel write_draft nodes.
    return {"blog_drafts": [draft]}


def collect_drafts(state: NewsComposerState) -> dict:
    """Join node: log draft count after parallel fan-out; no state change needed."""
    drafts = state.get("blog_drafts", [])
    print(f"[collect_drafts] Collected {len(drafts)} drafts.", flush=True)
    return {}
