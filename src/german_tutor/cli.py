"""Terminal entry point for the German tutor (LangGraph edition).

Two persistence layers:
  1. Conversation + graph state -> LangGraph SqliteSaver checkpointer, keyed by
     thread_id = the learner id. Re-running with the same id resumes the exact
     graph state (message history, in-flight lesson), which is what makes the
     tutor "remember" the previous session.
  2. Durable learning progress -> german_tutor.persistence.Store (separate db):
     level, per-unit mastery, lesson pointer, spaced-repetition deck, errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_tutor_graph
from .logging import ConversationLog, next_numbered_log_file
from .persistence import DEFAULT_DB_PATH, Store
from .tts import play, synthesize, tts_available

_DE_MARKER = re.compile(r"\[\[de:(.+?)\]\]", re.DOTALL)

CHECKPOINT_DB = ".german_tutor/graph.db"

COMMAND_PROMPTS = {
    "/lesson": "Start or continue my lesson for the next unit.",
    "/resume": "Resume my previous lesson exactly where I left off.",
    "/review": "Let's review my vocabulary that is due today.",
    "/quiz": "Give me a quiz of exercises for my current unit.",
    "/practice": "Let's have a short German conversation at my level.",
    "/placement": "Please run a placement check and set my CEFR level.",
    "/progress": "How am I doing? Summarize my progress.",
}


def help_text() -> str:
    return "\n".join(
        [
            "Commands:",
            "  /lesson      Start or continue a lesson",
            "  /resume      Resume your previous lesson",
            "  /review      Review vocabulary due today (spaced repetition)",
            "  /quiz        Practice exercises for your current unit",
            "  /practice    Have a German conversation",
            "  /placement   Take a placement check to set your level",
            "  /progress    See how you're doing",
            "  /play N      Play the Nth 🔊 word from the last reply aloud",
            "  /level <A1|A2|B1|B2>   Set your level directly",
            "  /whoami      Show your learner id",
            "  /help        Show this help",
            "  /quit /exit  Leave (your progress is saved)",
        ]
    )


def _osc8(uri: str, label: str) -> str:
    """Render an OSC 8 terminal hyperlink (clickable in iTerm2/VS Code/WezTerm/Kitty;
    shown as plain text in terminals without OSC 8 support, e.g. Apple Terminal.app)."""
    esc = "\x1b"
    return f"{esc}]8;;{uri}{esc}\\{label}{esc}]8;;{esc}\\"


def render_audio_markers(text: str) -> tuple[str, list[Path]]:
    """Replace [[de:WORD]] markers with the German text + a numbered 🔊 control.

    The 🔊N is a clickable OSC 8 link (plays in iTerm2/WezTerm/Kitty) AND is numbered
    so `/play N` works everywhere (VS Code, Apple Terminal) via afplay. Returns the
    rendered text and the ordered list of audio file paths for `/play`.
    """
    available = tts_available()
    paths: list[Path] = []

    def repl(match: re.Match) -> str:
        german = match.group(1).strip()
        if not available:
            return german
        path = synthesize(german)
        if path is None or not path.exists():
            return german
        paths.append(path)
        n = len(paths)
        return f"{german} {_osc8(path.resolve().as_uri(), f'🔊{n}')}"

    rendered = _DE_MARKER.sub(repl, text)
    return rendered, paths


def _turn_sources(messages, since: int) -> list[str]:
    """Collect 'source (license)' strings from get_lesson_material tool results
    produced in this turn (messages added after index `since`)."""
    sources: list[str] = []
    for msg in messages[since:]:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else ""
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and data.get("source"):
            tag = f"{data['source']}" + (f" ({data['license']})" if data.get("license") else "")
            if tag not in sources:
                sources.append(tag)
    return sources


def run_turn(graph, user_input: str, config: dict) -> tuple[str, list[str]]:
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    messages = result["messages"]
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    since = human_idxs[-1] if human_idxs else 0
    sources = _turn_sources(messages, since)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content, sources
    return "(no response)", sources


def chat(args: argparse.Namespace) -> int:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required. See .env.example.")
        return 2

    learner_id = args.learner_id
    store = Store(args.db)
    profile = store.get_or_create_learner(learner_id, name=args.name)
    config = {"configurable": {"thread_id": f"learner-{learner_id}"}}

    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else next_numbered_log_file()
    conversation_log = ConversationLog(log_path, learner_id=learner_id)

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_tutor_graph(store, learner_id, audio_markup=True).compile(
            checkpointer=checkpointer
        )

        print("Deutsch-Tutor — CEFR A1-B2 German learning (LangGraph)")
        print(f"Learner: {learner_id}")
        print(f"Log: {log_path}")
        if store.is_returning(learner_id):
            print(f"Willkommen zurück! {store.welcome_back(learner_id)}")
        else:
            print(
                f"Current level: {profile['current_level']}. "
                "Tip: run /placement to find your level."
            )
        print("Type /help for commands.\n")

        last_audio: list[Path] = []

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nTschüss! Your progress is saved.")
                store.close()
                return 0

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"/quit", "/exit"}:
                print("Tschüss! Your progress is saved.")
                store.close()
                return 0
            if lowered == "/help":
                print(help_text())
                continue
            if lowered.startswith("/play"):
                parts = user_input.split()
                n = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
                if not last_audio:
                    print("No audio from the last message yet.")
                elif not (1 <= n <= len(last_audio)):
                    print(f"Pick 1–{len(last_audio)} (e.g. /play 1).")
                elif not play(last_audio[n - 1]):
                    print("Could not play audio (no afplay).")
                continue
            if lowered == "/whoami":
                print(f"Learner id: {learner_id} (level {store.get_level(learner_id)})")
                continue
            if lowered.startswith("/level"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].upper() in ("A1", "A2", "B1", "B2"):
                    store.set_level(learner_id, parts[1].upper())
                    print(f"Level set to {parts[1].upper()}.")
                else:
                    print("Usage: /level A1|A2|B1|B2")
                continue

            prompt = COMMAND_PROMPTS.get(lowered, user_input)
            try:
                reply, sources = run_turn(graph, prompt, config)
                rendered, last_audio = render_audio_markers(reply)
                print(f"\nTutor: {rendered}")
                if last_audio:
                    print("  (click 🔊, or type /play N to hear a word)")
                for src in sources:
                    print(f"  📖 Source: {src}")
                print()
                conversation_log.append_round(user=user_input, assistant=reply)
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive on errors
                print(f"\n[error] {exc}\n")
                conversation_log.append_round(user=user_input, assistant=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent CEFR German tutor (A1-B2), LangGraph.")
    parser.add_argument(
        "--learner-id",
        default=os.getenv("TUTOR_LEARNER_ID", "default"),
        help="Stable id used to persist progress and conversation. Default: 'default'.",
    )
    parser.add_argument("--name", default=None, help="Display name for a new learner.")
    parser.add_argument(
        "--db",
        default=os.getenv("TUTOR_DB", str(DEFAULT_DB_PATH)),
        help="Path to the progress SQLite database.",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("TUTOR_LOG_FILE"),
        help="Conversation log path. Defaults to the next logs/conversation-NNN.json.",
    )
    return parser


def main() -> int:
    return chat(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
