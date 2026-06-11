"""FastAPI backend for the German tutor web GUI.

Wraps the existing LangGraph tutor (audio_markup=True so German is wrapped in
[[de:...]] markers the frontend turns into clickable 🔊 buttons), persists
conversation state with the same SQLite checkpointer keyed by learner id, and
serves on-demand pronunciation audio.

Run:  uv run german-tutor-web    (then open http://127.0.0.1:8000)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..graph import build_tutor_graph
from ..logging import ConversationLog, next_numbered_log_file
from ..persistence import DEFAULT_DB_PATH, Store
from ..tts import synthesize, tts_available

STATIC_DIR = Path(__file__).parent / "static"
CHECKPOINT_DB = os.getenv("TUTOR_CHECKPOINT_DB", ".german_tutor/graph.db")
PROGRESS_DB = os.getenv("TUTOR_DB", str(DEFAULT_DB_PATH))

COMMAND_PROMPTS = {
    "lesson": "Start or continue my lesson for the next unit.",
    "resume": "Resume my previous lesson exactly where I left off.",
    "review": "Let's review my vocabulary that is due today.",
    "quiz": "Give me a quiz of exercises for my current unit.",
    "practice": "Let's have a short German conversation at my level.",
    "placement": "Please run a placement check and set my CEFR level.",
    "progress": "How am I doing? Summarize my progress.",
}


class ChatRequest(BaseModel):
    learner_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    store = Store(PROGRESS_DB)
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        app.state.store = store
        app.state.saver = saver
        app.state.graphs = {}  # learner_id -> compiled graph (cached)
        app.state.logs = {}    # learner_id -> ConversationLog (one file per learner)
        yield
    store.close()


def _log_for(learner_id: str) -> ConversationLog:
    logs = app.state.logs
    if learner_id not in logs:
        path = next_numbered_log_file()
        logs[learner_id] = ConversationLog(path, learner_id=learner_id)
        print(f"[web] logging learner '{learner_id}' to {path}")
    return logs[learner_id]


app = FastAPI(title="German Tutor", lifespan=lifespan)


def _graph_for(learner_id: str):
    graphs = app.state.graphs
    if learner_id not in graphs:
        app.state.store.get_or_create_learner(learner_id)
        graphs[learner_id] = build_tutor_graph(
            app.state.store, learner_id, audio_markup=True
        ).compile(checkpointer=app.state.saver)
    return graphs[learner_id]


def _run_turn(learner_id: str, message: str) -> str:
    graph = _graph_for(learner_id)
    config = {"configurable": {"thread_id": f"learner-{learner_id}"}}
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "(no response)"


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY is not set on the server.")
    message = COMMAND_PROMPTS.get(req.message.strip().lower(), req.message)
    reply = await run_in_threadpool(_run_turn, req.learner_id, message)
    _log_for(req.learner_id).append_round(user=req.message, assistant=reply)
    return ChatResponse(reply=reply)


@app.get("/api/state")
async def state(learner_id: str = "default"):
    store: Store = app.state.store
    store.get_or_create_learner(learner_id)
    return {
        "learner_id": learner_id,
        "returning": store.is_returning(learner_id),
        "summary": store.welcome_back(learner_id),
        "level": store.get_level(learner_id),
    }


@app.get("/api/tts")
async def tts(text: str, voice: str | None = None):
    if not tts_available():
        raise HTTPException(503, "TTS not available on this server (no macOS 'say').")
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    path = await run_in_threadpool(synthesize, text, voice)
    if path is None:
        raise HTTPException(500, "synthesis failed")
    media = "audio/wav" if path.suffix == ".wav" else "audio/aiff"
    return FileResponse(str(path), media_type=media)


# Serve the SPA last so /api/* wins.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> int:
    import uvicorn

    host = os.getenv("TUTOR_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("TUTOR_WEB_PORT", "8000"))
    print(f"German Tutor web GUI → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
