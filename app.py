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

# ── Pipeline definition ───────────────────────────────────────────────────────
PIPELINE_STEPS = [
    {
        "icon": "🔍",
        "name": "Fetch Sources",
        "description": "Pulls articles from RSS feeds, Hacker News and Reddit in parallel",
        "tags": ["feedparser", "aiohttp", "BeautifulSoup", "HN Algolia API", "Reddit JSON API"],
        "nodes": {"fetch_sources"},
    },
    {
        "icon": "🏆",
        "name": "Rank & Filter",
        "description": "LLM scores each article 0–10 for relevance to configured topics, keeps top 20",
        "tags": ["LangChain", "OpenRouter", "GPT-4o-mini"],
        "nodes": {"rank_and_filter"},
    },
    {
        "icon": "⚡",
        "name": "Summarise + Compile Links",
        "description": "Two LLM branches run in parallel — top 5 summaries and trending link picks",
        "tags": ["LangGraph parallel edges", "LangChain", "OpenRouter"],
        "nodes": {"summarize", "compile_links", "join_digest"},
    },
    {
        "icon": "✍️",
        "name": "Draft Blog Posts",
        "description": "Fan-out to 3 simultaneous LLM writers: Opinion, Newsletter Recap, Deep-Dive",
        "tags": ["LangGraph Send API", "map-reduce", "OpenRouter"],
        "nodes": {"write_draft", "collect_drafts"},
    },
    {
        "icon": "👤",
        "name": "Human-in-the-Loop Review",
        "description": "Graph pauses — you approve or request changes on each draft before continuing",
        "tags": ["LangGraph interrupt()", "LangGraph Command(resume=)", "Streamlit"],
        "nodes": {"human_review", "rewrite_draft"},
    },
    {
        "icon": "💾",
        "name": "Finalise",
        "description": "Approved drafts written to dated markdown files, run state persisted to disk",
        "tags": ["LangGraph SqliteSaver", "SQLite", "Python"],
        "nodes": {"finalize"},
    },
    {
        "icon": "📧",
        "name": "Send Newsletter",
        "description": "Digest converted to styled HTML and emailed to recipients",
        "tags": ["Brevo API", "requests", "markdown"],
        "nodes": {"send_email"},
    },
]

TECH_STACK = [
    "Python", "LangGraph", "LangChain", "OpenRouter",
    "Streamlit", "SQLite", "Brevo API",
]


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
if "completed_nodes" not in st.session_state:
    st.session_state.completed_nodes = set()


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
    """
    graph = st.session_state.graph

    with st.status("Running pipeline...", expanded=True) as status:
        try:
            for event in graph.stream(command, config=config, stream_mode="updates"):
                node_name = list(event.keys())[0] if event else ""
                if node_name and node_name not in ("__start__", "__interrupt__"):
                    st.session_state.completed_nodes.add(node_name)
                    label = _node_display_name(node_name)
                    status.write(f"✓ {label}")
                    print(f"[graph] {node_name}", flush=True)
            status.update(label="Pipeline run complete", state="complete")
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


def _node_display_name(node_name: str) -> str:
    return {
        "fetch_sources": "Fetch Sources",
        "rank_and_filter": "Rank & Filter",
        "summarize": "Summarise",
        "compile_links": "Compile Links",
        "join_digest": "Join Digest",
        "write_draft": "Write Draft",
        "collect_drafts": "Collect Drafts",
        "human_review": "Human Review",
        "rewrite_draft": "Rewrite Draft",
        "finalize": "Finalise",
        "send_email": "Send Newsletter",
    }.get(node_name, node_name)


# ── UI helpers ────────────────────────────────────────────────────────────────
def _badge(text: str, color: str = "#eef0f5", text_color: str = "#444") -> str:
    return (
        f'<span style="background:{color};color:{text_color};border-radius:4px;'
        f'padding:2px 8px;font-size:0.72rem;margin:2px 2px 2px 0;display:inline-block;'
        f'border:1px solid #dde;">{text}</span>'
    )


def _render_pipeline_sidebar(completed_nodes: set) -> None:
    for step in PIPELINE_STEPS:
        done = bool(step["nodes"] & completed_nodes)
        is_active = (
            st.session_state.graph_status == "interrupted"
            and "human_review" in step["nodes"]
        )
        if done:
            border_color = "#b8dfc8"
            bg_color = "#f0fff6"
            status_icon = "✅"
        elif is_active:
            border_color = "#f0ad4e"
            bg_color = "#fffbf0"
            status_icon = "🟠"
        else:
            border_color = "#e2e6ea"
            bg_color = "#fafbfc"
            status_icon = "⬜"

        badges_html = "".join(_badge(t) for t in step["tags"])

        st.markdown(
            f"""
            <div style="margin-bottom:10px;padding:10px 12px;border-radius:8px;
                        border:1px solid {border_color};background:{bg_color};">
              <div style="font-weight:600;font-size:0.9rem;margin-bottom:3px;">
                {status_icon} {step['icon']} {step['name']}
              </div>
              <div style="font-size:0.78rem;color:#666;margin-bottom:6px;line-height:1.4;">
                {step['description']}
              </div>
              <div>{badges_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>📰 Agentic News Composer</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666;margin-top:4px;font-size:1rem;'>"
    "An agentic LLM pipeline that fetches news, ranks it, drafts blog posts, "
    "and emails a newsletter — with a human-in-the-loop review step."
    "</p>",
    unsafe_allow_html=True,
)

# Tech stack pills
stack_html = "".join(
    _badge(t, color="#dbe9ff", text_color="#1a4fa0") for t in TECH_STACK
)
st.markdown(
    f'<div style="margin-bottom:1.5rem;">Tech stack: {stack_html}</div>',
    unsafe_allow_html=True,
)

st.divider()

col_pipeline, col_main = st.columns([2, 3], gap="large")

# ── Left: Pipeline overview ───────────────────────────────────────────────────
with col_pipeline:
    st.markdown("### Pipeline")
    st.caption("Each step lights up as it completes.")
    pipeline_placeholder = st.empty()

with pipeline_placeholder.container():
    _render_pipeline_sidebar(st.session_state.completed_nodes)

# ── Right: Main content ───────────────────────────────────────────────────────
with col_main:

    # Controls
    st.markdown("### Controls")
    status_label = {
        "idle": "🟡 Idle — ready to run",
        "interrupted": "🟠 Waiting for your review below",
        "done": "🟢 Done",
        "error": "🔴 Error",
    }.get(st.session_state.graph_status, "")
    st.markdown(f"**Status:** {status_label}")

    if st.session_state.graph_status in ("idle", "done", "error"):
        if st.button("▶ Run Today's Digest", type="primary"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.graph_status = "idle"
            st.session_state.interrupt_data = None
            st.session_state.feedback = {}
            st.session_state.completed_nodes = set()
            _save_current_run(st.session_state.thread_id)

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            _stream_until_interrupt({}, config)
            st.rerun()

    # Digest view
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
            st.divider()
            st.markdown("### Today's Digest")
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

    # Human review
    interrupt_data = st.session_state.interrupt_data
    if st.session_state.graph_status == "interrupted" and isinstance(interrupt_data, dict):
        st.divider()
        st.markdown("### Review Blog Drafts")
        st.info(
            "The pipeline is paused waiting for your input. "
            "Approve or request changes on each draft, then submit to resume."
        )

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

    # Done
    if st.session_state.graph_status == "done":
        today = date.today().isoformat()
        digest_files = sorted(Path("output").glob(f"{today}-*-digest.md"))
        st.divider()
        if digest_files:
            st.success(f"Digest saved to `{digest_files[-1]}`")

            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                snap = st.session_state.graph.get_state(config)
                if snap and snap.values.get("email_sent"):
                    st.info("📧 Newsletter emailed to recipients.")
            except Exception:
                pass

            with st.expander("View digest"):
                st.markdown(digest_files[-1].read_text())
            for suffix, label in [("summaries", "summaries"), ("articles", "ranked articles")]:
                files = sorted(Path("output").glob(f"{today}-*-{suffix}.md"))
                if files:
                    with st.expander(f"View {label}"):
                        st.markdown(files[-1].read_text())
        else:
            st.success("Digest finalized! Check the `output/` folder.")

    # Error
    if st.session_state.graph_status == "error":
        st.divider()
        st.error("The pipeline encountered an error.")
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
                st.session_state.completed_nodes = set()
                _CURRENT_RUN_FILE.unlink(missing_ok=True)
                st.rerun()
