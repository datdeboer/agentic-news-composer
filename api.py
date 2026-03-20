"""FastAPI backend for the Agentic News Composer."""
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Agentic News Composer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Graph singleton ───────────────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from graph.graph import get_compiled_graph
        _graph, _ = get_compiled_graph()
    return _graph


def _sse(event: str, **data) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


def _stream_graph(command, config: dict):
    """Shared generator: stream node completions then emit interrupted/done."""
    graph = get_graph()

    for event in graph.stream(command, config=config, stream_mode="updates"):
        node_name = list(event.keys())[0] if event else ""
        if node_name and node_name not in ("__start__", "__interrupt__"):
            yield _sse("node_complete", node=node_name)

    snap = graph.get_state(config)
    for task in snap.tasks or []:
        if task.interrupts:
            yield _sse("interrupted", drafts=snap.values.get("blog_drafts", []))
            return

    state = snap.values
    yield _sse("done", email_sent=state.get("email_sent", False))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/providers")
def get_providers():
    from graph.llm import PROVIDERS
    return PROVIDERS


class RunBody(BaseModel):
    provider: str = "openrouter"
    model: str | None = None


@app.post("/api/runs")
def create_run(body: RunBody):
    import graph.llm as llm_module
    model = body.model or llm_module.PROVIDERS[body.provider]["models"][0]["id"]
    llm_module.configure(body.provider, model)
    return {"thread_id": str(uuid.uuid4())}


@app.get("/api/runs/{thread_id}/stream")
def stream_run(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return StreamingResponse(
        _stream_graph({}, config),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/runs/{thread_id}/state")
def get_state(thread_id: str):
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)
    if not snap:
        return {}
    return {
        "top_5_summaries": snap.values.get("top_5_summaries", []),
        "top_5_links": snap.values.get("top_5_links", []),
        "blog_drafts": snap.values.get("blog_drafts", []),
        "email_sent": snap.values.get("email_sent", False),
    }


class ReviewBody(BaseModel):
    feedback: list[dict]


@app.post("/api/runs/{thread_id}/review")
def submit_review(thread_id: str, body: ReviewBody):
    from langgraph.types import Command
    config = {"configurable": {"thread_id": thread_id}}
    return StreamingResponse(
        _stream_graph(Command(resume=body.feedback), config),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Serve built React app (production) ───────────────────────────────────────
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
