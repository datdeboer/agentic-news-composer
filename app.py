"""Streamlit UI for the Agentic News Composer."""
import threading
import time
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import get_compiled_graph
from langgraph.types import Command

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic News Composer",
    page_icon="📰",
    layout="wide",
)

# ── Module-level shared state (safe to write from background threads) ─────────
# st.session_state is not accessible from background threads, so we use a plain
# dict here and sync it into session_state on each Streamlit rerender.
_shared: dict = {}
_shared_lock = threading.Lock()


def _shared_get(key, default=None):
    with _shared_lock:
        return _shared.get(key, default)


def _shared_set(key, value):
    with _shared_lock:
        _shared[key] = value


def _shared_append(key, value):
    with _shared_lock:
        _shared.setdefault(key, []).append(value)


# ── Session state defaults ───────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "graph_status" not in st.session_state:
    st.session_state.graph_status = "idle"  # idle | running | interrupted | done | error
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None
if "graph" not in st.session_state:
    try:
        st.session_state.graph, st.session_state.checkpointer = get_compiled_graph()
        print("[app] Graph compiled and checkpointer ready", flush=True)
    except Exception as e:
        st.error(f"Failed to initialize graph: {e}")
        st.stop()
if "run_log" not in st.session_state:
    st.session_state.run_log = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}


def _sync_from_shared():
    """Pull updates written by background threads into session_state."""
    for key in ("graph_status", "interrupt_data"):
        val = _shared_get(key)
        if val is not None:
            setattr(st.session_state, key, val)
            with _shared_lock:
                _shared.pop(key, None)
    log_entries = _shared_get("run_log_pending", [])
    if log_entries:
        st.session_state.run_log.extend(log_entries)
        with _shared_lock:
            _shared["run_log_pending"] = []


# ── Helper: run graph in background thread ───────────────────────────────────
INTERRUPT_NODES = {"human_review"}  # nodes listed in interrupt_before


def _check_interrupt(graph, config: dict) -> bool:
    """Return True and update shared state if graph is paused at an interrupt node."""
    snap = graph.get_state(config)
    if snap and snap.next and snap.next[0] in INTERRUPT_NODES:
        # Collect interrupt value from tasks if present, else pull from state directly
        interrupt_value = None
        for task in (snap.tasks or []):
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
        if interrupt_value is None:
            # Fallback: surface the relevant state fields directly
            interrupt_value = {
                "drafts": snap.values.get("blog_drafts", []),
                "summaries": snap.values.get("top_5_summaries", []),
                "links": snap.values.get("top_5_links", []),
            }
        _shared_set("interrupt_data", interrupt_value)
        _shared_set("graph_status", "interrupted")
        return True
    return False


def _run_graph_thread(graph, thread_id: str, initial_state: dict):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        print("[app] Background thread started", flush=True)
        _shared_append("run_log_pending", "Graph started...")
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            node_name = list(event.keys())[0] if event else "unknown"
            if node_name not in ("__start__",):
                print(f"[app] Node completed: {node_name}", flush=True)
                _shared_append("run_log_pending", f"✓ {node_name}")
            if _check_interrupt(graph, config):
                print("[app] Graph interrupted — waiting for human review", flush=True)
                return

        # Stream ended — check one more time in case stream ended at interrupt
        if not _check_interrupt(graph, config):
            print("[app] Graph finished", flush=True)
            _shared_set("graph_status", "done")
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[app] ERROR in graph thread:\n{msg}", flush=True)
        _shared_append("run_log_pending", f"Error: {e}")
        _shared_append("run_log_pending", msg)
        _shared_set("graph_status", "error")


def _resume_graph_thread(graph, thread_id: str, feedback: list):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        print("[app] Resuming graph with feedback", flush=True)
        _shared_append("run_log_pending", "Resuming with feedback...")
        for event in graph.stream(Command(resume=feedback), config=config, stream_mode="updates"):
            node_name = list(event.keys())[0] if event else "unknown"
            if node_name not in ("__start__",):
                print(f"[app] Node completed: {node_name}", flush=True)
                _shared_append("run_log_pending", f"✓ {node_name}")
            if _check_interrupt(graph, config):
                print("[app] Graph interrupted again — waiting for review", flush=True)
                return

        if not _check_interrupt(graph, config):
            print("[app] Graph finished", flush=True)
            _shared_set("graph_status", "done")
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[app] ERROR in resume thread:\n{msg}", flush=True)
        _shared_append("run_log_pending", f"Error: {e}")
        _shared_append("run_log_pending", msg)
        _shared_set("graph_status", "error")


# ── Sync thread updates into session_state on every rerender ─────────────────
_sync_from_shared()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📰 Agentic News Composer")
st.caption("Powered by LangChain + LangGraph + OpenRouter")

# ── Section 1: Controls ───────────────────────────────────────────────────────
st.header("1. Controls")

col1, col2 = st.columns([3, 1])
with col1:
    status_badge = {
        "idle": "🟡 Idle",
        "running": "🔵 Running...",
        "interrupted": "🟠 Waiting for review — scroll down to Section 3",
        "done": "🟢 Done",
        "error": "🔴 Error",
    }.get(st.session_state.graph_status, "")
    st.markdown(f"**Status:** {status_badge}")
    if st.session_state.graph_status not in ("running",):
        if st.button("Refresh", help="Sync latest status from the graph"):
            st.rerun()

with col2:
    run_disabled = st.session_state.graph_status in ("running", "interrupted")
    if st.button("Run Today's Digest", disabled=run_disabled, type="primary"):
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id
        st.session_state.graph_status = "running"
        st.session_state.interrupt_data = None
        st.session_state.run_log = []
        st.session_state.feedback = {}

        t = threading.Thread(
            target=_run_graph_thread,
            args=(st.session_state.graph, thread_id, {}),
            daemon=True,
        )
        t.start()
        st.rerun()

# Auto-refresh while running
if st.session_state.graph_status == "running":
    time.sleep(2)
    st.rerun()

# Run log — shown inline while running, collapsed expander when done
if st.session_state.run_log:
    if st.session_state.graph_status == "running":
        st.markdown("**Progress**")
        log_box = st.empty()
        log_box.code("\n".join(st.session_state.run_log), language=None)
    else:
        with st.expander("Run log", expanded=False):
            st.code("\n".join(st.session_state.run_log), language=None)

# ── Section 2: Digest view ────────────────────────────────────────────────────
interrupt_data = st.session_state.interrupt_data
graph_state = None
if st.session_state.thread_id and st.session_state.graph_status in ("interrupted", "done"):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snap = st.session_state.graph.get_state(config)
    graph_state = snap.values if snap else None

if graph_state:
    st.header("2. Today's Digest")

    summaries = graph_state.get("top_5_summaries", [])
    links = graph_state.get("top_5_links", [])

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
if st.session_state.graph_status == "interrupted" and interrupt_data:
    st.header("3. Review Blog Drafts")
    st.info("Review each draft below. Approve or request changes. Submit when all 3 are actioned.")

    drafts = interrupt_data.get("drafts", [])
    style_labels = {"opinion": "Opinion / Editorial", "newsletter": "Newsletter Recap", "deep-dive": "Deep Dive"}

    all_actioned = True
    feedback_list = []

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

                # Determine action
                if approved:
                    action = "approve"
                elif notes.strip():
                    action = "revise"
                else:
                    all_actioned = False
                    action = None

                if action:
                    feedback_list.append({"style": style, "action": action, "notes": notes})
                else:
                    feedback_list.append({"style": style, "action": None, "notes": ""})

        # Check all actioned
        all_actioned = all(f["action"] is not None for f in feedback_list)

        if st.button(
            "Submit Review",
            disabled=not all_actioned,
            type="primary",
            help="Approve or request changes on all 3 drafts to enable submission.",
        ):
            st.session_state.feedback = {f["style"]: f for f in feedback_list}
            st.session_state.graph_status = "running"
            st.session_state.interrupt_data = None
            t = threading.Thread(
                target=_resume_graph_thread,
                args=(st.session_state.graph, st.session_state.thread_id, feedback_list),
                daemon=True,
            )
            t.start()
            st.rerun()

        if not all_actioned:
            st.warning("Please approve or request changes for all 3 drafts before submitting.")

# ── Done state ─────────────────────────────────────────────────────────────────
if st.session_state.graph_status == "done":
    today = date.today().isoformat()
    # Find the most recently written digest file for today
    output_files = sorted(Path("output").glob(f"{today}-*-digest.md"))
    if output_files:
        output_path = output_files[-1]
        st.success(f"Digest finalized! Saved to `{output_path}`")
        with st.expander("View digest"):
            st.markdown(output_path.read_text())
        summary_files = sorted(Path("output").glob(f"{today}-*-summaries.md"))
        article_files = sorted(Path("output").glob(f"{today}-*-articles.md"))
        if summary_files:
            with st.expander("View summaries"):
                st.markdown(summary_files[-1].read_text())
        if article_files:
            with st.expander("View ranked articles"):
                st.markdown(article_files[-1].read_text())
    else:
        st.success("Digest finalized! Check the `output/` folder.")

if st.session_state.graph_status == "error":
    st.error("Graph encountered an error. Check the run log above.")
    if st.button("Reset"):
        st.session_state.graph_status = "idle"
        st.session_state.thread_id = None
        st.session_state.interrupt_data = None
        st.rerun()
