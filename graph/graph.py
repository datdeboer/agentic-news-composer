"""LangGraph graph definition for the Agentic News Composer."""
import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.nodes.compile_links import compile_links
from graph.nodes.draft_blog_posts import collect_drafts, fan_out_drafts, write_draft
from graph.nodes.fetch_sources import fetch_sources
from graph.nodes.finalize import finalize
from graph.nodes.human_review import human_review
from graph.nodes.rank_and_filter import rank_and_filter
from graph.nodes.regenerate_drafts import rewrite_draft
from graph.nodes.summarize import summarize
from graph.state import NewsComposerState

DB_PATH = os.environ.get("CHECKPOINTER_DB", "output/checkpoints.db")


def _route_after_review(state: NewsComposerState):
    """
    Conditional edge after human_review.

    Returns "finalize" if all drafts approved, or list[Send] to rewrite_draft
    for each flagged draft (revision cycle).
    """
    feedback = state.get("human_feedback") or []
    print(f"[route] Received feedback: {feedback}", flush=True)

    if not feedback:
        print("[route] No feedback in state — routing to finalize as fallback.", flush=True)
        return "finalize"

    to_revise = {f["style"]: f.get("notes", "") for f in feedback if f.get("action") == "revise"}

    if not to_revise:
        print("[route] All approved → finalize.", flush=True)
        return "finalize"

    approved_drafts = [
        d for d in state.get("blog_drafts", [])
        if d["style"] not in to_revise
    ]
    print(f"[route] Revising {list(to_revise.keys())}, keeping {[d['style'] for d in approved_drafts]}", flush=True)

    # Pass only plain state values (no annotated keys) to avoid reducer conflicts in Send
    plain_state = {
        "topics": state.get("topics", []),
        "raw_articles": state.get("raw_articles", []),
        "top_5_summaries": state.get("top_5_summaries", []),
        "top_5_links": state.get("top_5_links", []),
        "human_feedback": state.get("human_feedback"),
        "finalized": state.get("finalized", False),
    }
    return [
        Send("rewrite_draft", {
            **plain_state,
            "draft_style": style,
            "revision_notes": notes,
            "blog_drafts": approved_drafts,
        })
        for style, notes in to_revise.items()
    ]


def build_graph(checkpointer=None):
    """
    Build and compile the NewsComposer LangGraph graph.

    Graph flow:
        fetch_sources
            → rank_and_filter
            → [summarize | compile_links]  (parallel fan-out)
            → join_digest
            → fan_out_drafts  (Send API → 3x write_draft)
            → collect_drafts
            → human_review  (interrupt)
            → _route_after_review (conditional)
                "finalize" → finalize → END
                list[Send] → rewrite_draft → collect_drafts → human_review  (cycle)
    """
    graph = StateGraph(NewsComposerState)

    # --- Nodes ---
    graph.add_node("fetch_sources", fetch_sources)
    graph.add_node("rank_and_filter", rank_and_filter)
    graph.add_node("summarize", summarize)
    graph.add_node("compile_links", compile_links)
    graph.add_node("join_digest", _join_digest)
    graph.add_node("write_draft", write_draft)
    graph.add_node("collect_drafts", collect_drafts)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)
    graph.add_node("rewrite_draft", rewrite_draft)

    # --- Edges ---
    graph.add_edge(START, "fetch_sources")
    graph.add_edge("fetch_sources", "rank_and_filter")

    # Parallel: rank_and_filter → summarize + compile_links
    graph.add_edge("rank_and_filter", "summarize")
    graph.add_edge("rank_and_filter", "compile_links")

    # Join: both branches → join_digest
    graph.add_edge("summarize", "join_digest")
    graph.add_edge("compile_links", "join_digest")

    # Send API fan-out → 3x write_draft
    graph.add_conditional_edges("join_digest", fan_out_drafts, ["write_draft"])
    graph.add_edge("write_draft", "collect_drafts")
    graph.add_edge("collect_drafts", "human_review")

    # Conditional routing after review: finalize or revision cycle
    graph.add_conditional_edges(
        "human_review",
        _route_after_review,
        ["finalize", "rewrite_draft"],
    )

    # Revision cycle: rewrite → collect → human_review (loops until all approved)
    graph.add_edge("rewrite_draft", "collect_drafts")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def get_checkpointer():
    """Return a SqliteSaver checkpointer backed by a local SQLite database."""
    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    # check_same_thread=False is required — checkpointer is created in the main
    # thread but written to by background threads during graph execution.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def get_compiled_graph():
    """Return a compiled graph + checkpointer for use in Streamlit."""
    checkpointer = get_checkpointer()
    return build_graph(checkpointer=checkpointer), checkpointer


def _join_digest(state: NewsComposerState) -> dict:
    """No-op join node — LangGraph merges parallel branch outputs automatically."""
    print("[join_digest] Digest branches joined.")
    return {}
