"""Streamlit UI for the Agentic News Composer."""
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import get_compiled_graph
from langgraph.types import Command

st.set_page_config(
    page_title="Agentic News Composer",
    page_icon="📰",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "graph_status" not in st.session_state:
    st.session_state.graph_status = "idle"  # idle | interrupted | done | error
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None
if "graph" not in st.session_state:
    try:
        st.session_state.graph, _ = get_compiled_graph()
    except Exception as e:
        st.error(f"Failed to initialize graph: {e}")
        st.stop()
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ── Session recovery (survives browser refresh) ───────────────────────────────
_CURRENT_RUN_FILE = Path("output/.current_run")


def _save_current_run(thread_id: str) -> None:
    _CURRENT_RUN_FILE.parent.mkdir(exist_ok=True)
    _CURRENT_RUN_FILE.write_text(thread_id)


def _restore_session_if_needed() -> None:
    """On a fresh session, recover an interrupted run from the checkpointer."""
    if st.session_state.graph_status != "idle":
        return
    if not _CURRENT_RUN_FILE.exists():
        return
    thread_id = _CURRENT_RUN_FILE.read_text().strip()
    if not thread_id:
        return
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snap = st.session_state.graph.get_state(config)
    except Exception:
        return
    if not snap:
        return
    for task in snap.tasks or []:
        if task.interrupts:
            st.session_state.thread_id = thread_id
            st.session_state.graph_status = "interrupted"
            st.session_state.interrupt_data = task.interrupts[0].value
            print(f"[app] Recovered interrupted run: {thread_id}", flush=True)
            return


_restore_session_if_needed()


# ── Graph runner ──────────────────────────────────────────────────────────────
def _stream_until_interrupt(command, config: dict) -> None:
    """
    Run the graph synchronously, show live progress via st.status(), then
    update session state based on whether the graph hit an interrupt or finished.

    This runs in the main Streamlit thread — no background threads needed.
    st.status() streams node completions to the UI in real time.
    After this function returns, call st.rerun() to render the new state.
    """
    graph = st.session_state.graph

    with st.status("Running...", expanded=True) as status:
        try:
            for event in graph.stream(command, config=config, stream_mode="updates"):
                node_name = list(event.keys())[0] if event else ""
                if node_name and node_name not in ("__start__", "__interrupt__"):
                    status.write(f"✓ {node_name}")
                    print(f"[graph] {node_name}", flush=True)
            status.update(label="Done", state="complete")
        except Exception as e:
            import traceback
            status.update(label="Error", state="error")
            st.session_state.graph_status = "error"
            st.session_state.interrupt_data = traceback.format_exc()
            print(f"[graph] ERROR: {e}", flush=True)
            return

    snap = graph.get_state(config)
    for task in snap.tasks or []:
        if task.interrupts:
            st.session_state.graph_status = "interrupted"
            st.session_state.interrupt_data = task.interrupts[0].value
            print("[graph] Interrupted — waiting for human review", flush=True)
            return

    st.session_state.graph_status = "done"
    _CURRENT_RUN_FILE.unlink(missing_ok=True)
    print("[graph] Finished", flush=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📰 Agentic News Composer")
st.caption("Powered by LangChain + LangGraph + OpenRouter")

# ── Section 1: Controls ───────────────────────────────────────────────────────
st.header("1. Controls")

status_label = {
    "idle": "🟡 Idle",
    "interrupted": "🟠 Waiting for review — see Section 3 below",
    "done": "🟢 Done",
    "error": "🔴 Error",
}.get(st.session_state.graph_status, "")
st.markdown(f"**Status:** {status_label}")

if st.session_state.graph_status in ("idle", "done", "error"):
    if st.button("Run Today's Digest", type="primary"):
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id
        st.session_state.graph_status = "idle"
        st.session_state.interrupt_data = None
        st.session_state.feedback = {}
        _save_current_run(thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        _stream_until_interrupt({}, config)
        st.rerun()

# ── Section 2: Digest view ────────────────────────────────────────────────────
if st.session_state.thread_id and st.session_state.graph_status in ("interrupted", "done"):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        snap = st.session_state.graph.get_state(config)
        graph_state = snap.values if snap else {}
    except Exception:
        graph_state = {}

    summaries = graph_state.get("top_5_summaries", [])
    links = graph_state.get("top_5_links", [])

    if summaries or links:
        st.header("2. Today's Digest")
        col_s, col_l = st.columns(2)

        with col_s:
            st.subheader("Top 5 Stories")
            for i, s in enumerate(summaries, 1):
                with st.expander(f"{i}. {s.get('title', 'Untitled')}"):
                    st.markdown(s.get("summary", ""))
                    st.markdown(f"[Read more]({s.get('url', '#')})")

        with col_l:
            st.subheader("Trending Links")
            for i, l in enumerate(links, 1):
                st.markdown(f"**{i}.** [{l.get('title', '')}]({l.get('url', '#')})")
                st.caption(l.get("reason", ""))

# ── Section 3: Blog Drafts Review ─────────────────────────────────────────────
interrupt_data = st.session_state.interrupt_data
if st.session_state.graph_status == "interrupted" and isinstance(interrupt_data, dict):
    st.header("3. Review Blog Drafts")
    st.info("Approve or request changes on each draft, then click Submit.")

    drafts = interrupt_data.get("drafts", [])
    style_labels = {
        "opinion": "Opinion / Editorial",
        "newsletter": "Newsletter Recap",
        "deep-dive": "Deep Dive",
    }

    feedback_list = []
    all_actioned = True

    if drafts:
        tabs = st.tabs([style_labels.get(d["style"], d["style"]) for d in drafts])

        for tab, draft in zip(tabs, drafts):
            style = draft["style"]
            with tab:
                st.markdown(draft.get("content", ""))
                st.divider()

                col_a, col_r = st.columns([1, 3])
                with col_a:
                    approved = st.checkbox(
                        "Approve",
                        key=f"approve_{style}",
                        value=st.session_state.feedback.get(style, {}).get("action") == "approve",
                    )
                with col_r:
                    notes = st.text_area(
                        "Request changes (optional):",
                        key=f"notes_{style}",
                        value=st.session_state.feedback.get(style, {}).get("notes", ""),
                        height=80,
                    )

                if approved:
                    feedback_list.append({"style": style, "action": "approve", "notes": notes})
                elif notes.strip():
                    feedback_list.append({"style": style, "action": "revise", "notes": notes})
                else:
                    feedback_list.append({"style": style, "action": None, "notes": ""})
                    all_actioned = False

        if not all_actioned:
            st.warning("Please approve or request changes for all drafts before submitting.")

        if st.button(
            "Submit Review",
            disabled=not all_actioned,
            type="primary",
            help="Approve or request changes on all drafts to enable.",
        ):
            st.session_state.feedback = {f["style"]: f for f in feedback_list}
            st.session_state.interrupt_data = None

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            _stream_until_interrupt(Command(resume=feedback_list), config)
            st.rerun()

# ── Done ──────────────────────────────────────────────────────────────────────
if st.session_state.graph_status == "done":
    today = date.today().isoformat()
    digest_files = sorted(Path("output").glob(f"{today}-*-digest.md"))
    if digest_files:
        st.success(f"Digest finalized! Saved to `{digest_files[-1]}`")
        with st.expander("View digest"):
            st.markdown(digest_files[-1].read_text())
        for suffix, label in [("summaries", "summaries"), ("articles", "ranked articles")]:
            files = sorted(Path("output").glob(f"{today}-*-{suffix}.md"))
            if files:
                with st.expander(f"View {label}"):
                    st.markdown(files[-1].read_text())
    else:
        st.success("Digest finalized! Check the `output/` folder.")

# ── Error ─────────────────────────────────────────────────────────────────────
if st.session_state.graph_status == "error":
    st.error("Graph encountered an error.")
    if isinstance(st.session_state.interrupt_data, str):
        st.code(st.session_state.interrupt_data, language=None)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retry"):
            st.session_state.graph_status = "idle"
            st.rerun()
    with col2:
        if st.button("Reset"):
            st.session_state.graph_status = "idle"
            st.session_state.thread_id = None
            st.session_state.interrupt_data = None
            _CURRENT_RUN_FILE.unlink(missing_ok=True)
            st.rerun()
