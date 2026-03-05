"""LangGraph node: human-in-the-loop review using interrupt/resume."""
import logging

from langgraph.types import interrupt

from graph.state import NewsComposerState

logger = logging.getLogger(__name__)


def human_review(state: NewsComposerState) -> dict:
    """
    Interrupt graph execution and surface drafts + digest to the UI.

    The interrupt() call pauses the graph and saves state to the checkpointer.
    Streamlit polls the graph state, renders the review UI, then resumes via
    Command(resume=feedback).

    feedback shape:
        [{"style": "opinion", "action": "approve"|"revise", "notes": "..."}]
    """
    print("[human_review] Interrupting for human review...")

    feedback = interrupt({
        "drafts": state["blog_drafts"],
        "summaries": state["top_5_summaries"],
        "links": state["top_5_links"],
    })

    print(f"[human_review] Resumed with feedback: {feedback}")
    return {"human_feedback": feedback}


def should_finalize(state: NewsComposerState) -> str:
    """Conditional edge: route to finalize if all drafts approved, else regenerate."""
    feedback = state.get("human_feedback") or []
    if all(f.get("action") == "approve" for f in feedback):
        print("[should_finalize] All drafts approved → finalizing.")
        return "finalize"
    flagged = [f["style"] for f in feedback if f.get("action") == "revise"]
    print(f"[should_finalize] Drafts to revise: {flagged} → regenerating.")
    return "regenerate_drafts"
