from typing import Annotated, TypedDict


def _merge_drafts(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: merge draft lists by style, so parallel writes don't conflict.

    When multiple write_draft nodes run in parallel, each returns a single-item
    list. This reducer combines them into one list keyed by style, replacing any
    existing draft that shares the same style (used for rewrites too).
    """
    merged = {d["style"]: d for d in (existing or [])}
    for d in (new or []):
        merged[d["style"]] = d
    return list(merged.values())


class NewsComposerState(TypedDict):
    topics: list[str]
    raw_articles: list[dict]                          # {title, url, content, source, score}
    top_5_summaries: list[dict]                       # {title, url, summary}
    top_5_links: list[dict]                           # {title, url, reason}
    blog_drafts: Annotated[list[dict], _merge_drafts] # [{style, title, content}, x3]
    human_feedback: dict | None                       # [{style, action: approve|revise, notes}]
    finalized: bool
