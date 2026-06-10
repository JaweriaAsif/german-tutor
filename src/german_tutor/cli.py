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
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_tutor_graph
from .persistence import DEFAULT_DB_PATH, Store

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
            "  /level <A1|A2|B1|B2>   Set your level directly",
            "  /whoami      Show your learner id",
            "  /help        Show this help",
            "  /quit /exit  Leave (your progress is saved)",
        ]
    )


def run_turn(graph, user_input: str, config: dict) -> str:
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "(no response)"


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

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_tutor_graph(store, learner_id).compile(checkpointer=checkpointer)

        print("Deutsch-Tutor — CEFR A1-B2 German learning (LangGraph)")
        print(f"Learner: {learner_id}")
        if store.is_returning(learner_id):
            print(f"Willkommen zurück! {store.welcome_back(learner_id)}")
        else:
            print(
                f"Current level: {profile['current_level']}. "
                "Tip: run /placement to find your level."
            )
        print("Type /help for commands.\n")

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
                reply = run_turn(graph, prompt, config)
                print(f"\nTutor: {reply}\n")
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive on errors
                print(f"\n[error] {exc}\n")


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
    return parser


def main() -> int:
    return chat(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
