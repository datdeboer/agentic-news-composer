"""LangGraph node: rewrite a single flagged draft with revision notes."""
import logging
import os

from langchain_openai import ChatOpenAI

from graph.nodes.draft_blog_posts import STYLE_PROMPTS, _build_context

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )


def rewrite_draft(state: dict) -> dict:
    """
    Rewrite a single flagged draft incorporating reviewer notes.

    Expects state keys:
        draft_style: str        — the style to rewrite
        revision_notes: str     — reviewer's change request
        blog_drafts: list[dict] — currently approved drafts (without the revising one)
    """
    style = state["draft_style"]
    notes = state.get("revision_notes", "")
    print(f"[rewrite_draft:{style}] Rewriting with notes: {notes[:100]}")

    context = _build_context(state)
    style_prompt = STYLE_PROMPTS[style]

    prompt = f"""{style_prompt}

Here is today's news digest:
{context}

REVISION REQUEST — the reviewer asked for the following changes:
"{notes}"

Please address these notes in your revised version.
Return ONLY the revised blog post in markdown format, starting with a # title."""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content.strip()

    lines = content.split("\n")
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("#")), f"{style.title()} Draft")

    draft = {"style": style, "title": title, "content": content}
    print(f"[rewrite_draft:{style}] Done — '{title}'", flush=True)
    # Return only this draft — the _merge_drafts reducer replaces the old draft
    # with the same style in the accumulated state.
    return {"blog_drafts": [draft]}
