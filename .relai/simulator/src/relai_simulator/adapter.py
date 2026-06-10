from __future__ import annotations

import asyncio
import atexit
import os
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from german_tutor.cli import CHECKPOINT_DB, COMMAND_PROMPTS, help_text
from german_tutor.graph import build_tutor_graph
from german_tutor.persistence import DEFAULT_DB_PATH, Store
from german_tutor.tools_lc import make_tools
from relai_simulator.adapter_contract import (
    AgentAdapter,
    AgentTurnResult,
    ToolCallRecord,
    ToolResultRecord,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEARNER_ID = "relai-simulator"


class ProjectAgentAdapter:
    def __init__(self) -> None:
        self.learner_id = os.getenv("TUTOR_LEARNER_ID", DEFAULT_LEARNER_ID)
        self.db_path = self._resolve_path(os.getenv("TUTOR_DB"), DEFAULT_DB_PATH)
        self.thread_id = os.getenv(
            "RELAI_SIMULATOR_THREAD_ID",
            f"relai-simulator-{uuid.uuid4().hex}",
        )
        self.checkpoint_db = self._resolve_path(
            os.getenv("RELAI_SIMULATOR_CHECKPOINT_DB"),
            CHECKPOINT_DB,
        )
        self._message_count = 0
        self._store = Store(self.db_path)
        self._store.get_or_create_learner(self.learner_id)
        self._checkpointer_cm = SqliteSaver.from_conn_string(str(self.checkpoint_db))
        self._checkpointer = self._checkpointer_cm.__enter__()
        self._graph = build_tutor_graph(self._store, self.learner_id).compile(
            checkpointer=self._checkpointer
        )
        self._config = {"configurable": {"thread_id": self.thread_id}}
        self.agent_or_tools = self._build_agent_tools()
        atexit.register(self.close)

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        return await asyncio.to_thread(self._run_turn_sync, user_message)

    def close(self) -> None:
        checkpointer_cm = getattr(self, "_checkpointer_cm", None)
        if checkpointer_cm is not None:
            self._checkpointer_cm = None
            checkpointer_cm.__exit__(None, None, None)
        store = getattr(self, "_store", None)
        if store is not None:
            self._store = None
            store.close()

    def _run_turn_sync(self, user_message: str) -> AgentTurnResult:
        prepared_turn = self._prepare_turn(user_message)
        if isinstance(prepared_turn, AgentTurnResult):
            return prepared_turn
        user_message = prepared_turn

        result = self._graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=self._config,
        )
        messages = list(result.get("messages", []))
        new_messages = messages[self._message_count :] if self._message_count <= len(messages) else messages
        self._message_count = len(messages)

        tool_calls: list[ToolCallRecord] = []
        tool_results: list[ToolResultRecord] = []
        call_names: dict[str, str] = {}
        for message in new_messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    call_id = self._optional_str(tool_call.get("id"))
                    name = self._optional_str(tool_call.get("name")) or "tool"
                    if call_id is not None:
                        call_names[call_id] = name
                    tool_calls.append(
                        ToolCallRecord(
                            name=name,
                            arguments=tool_call.get("args", {}),
                            call_id=call_id,
                            metadata={"message_type": type(message).__name__},
                        )
                    )
            elif isinstance(message, ToolMessage):
                call_id = self._optional_str(getattr(message, "tool_call_id", None))
                tool_results.append(
                    ToolResultRecord(
                        name=getattr(message, "name", None) or call_names.get(call_id) or "tool",
                        result=self._message_content(message.content),
                        call_id=call_id,
                        metadata={"message_type": type(message).__name__},
                    )
                )

        assistant_message = self._extract_reply(messages)
        return AgentTurnResult(
            assistant_message=assistant_message,
            metadata={
                "learner_id": self.learner_id,
                "thread_id": self.thread_id,
            },
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    def _prepare_turn(self, user_message: str) -> str | AgentTurnResult:
        lowered = user_message.strip().lower()
        if lowered in COMMAND_PROMPTS:
            return COMMAND_PROMPTS[lowered]
        if lowered == "/help":
            return AgentTurnResult(assistant_message=help_text())
        if lowered == "/whoami":
            level = self._store.get_level(self.learner_id)
            return AgentTurnResult(
                assistant_message=f"Learner id: {self.learner_id} (level {level})"
            )
        if lowered.startswith("/level"):
            parts = user_message.split()
            if len(parts) == 2 and parts[1].upper() in {"A1", "A2", "B1", "B2"}:
                level = parts[1].upper()
                self._store.set_level(self.learner_id, level)
                return AgentTurnResult(assistant_message=f"Level set to {level}.")
            return AgentTurnResult(assistant_message="Usage: /level A1|A2|B1|B2")
        if lowered in {"/quit", "/exit"}:
            return AgentTurnResult(assistant_message="Tschüss! Your progress is saved.")
        return user_message

    def _build_agent_tools(self) -> list[object]:
        tools_by_name: dict[str, object] = {}
        for group in make_tools(self._store, self.learner_id).values():
            for tool in group:
                name = getattr(tool, "name", None)
                if isinstance(name, str) and name and name not in tools_by_name:
                    tools_by_name[name] = tool
        return list(tools_by_name.values())

    def _extract_reply(self, messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = self._message_content(message.content)
                if isinstance(content, str) and content.strip():
                    return content
        return "(no response)"

    @staticmethod
    def _message_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            normalized: list[object] = []
            for item in content:
                if isinstance(item, dict):
                    normalized.append(item)
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                else:
                    normalized.append(str(item))
            if text_parts:
                return "\n".join(part for part in text_parts if part.strip())
            return normalized
        return content

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _resolve_path(raw_value: str | os.PathLike[str] | None, default_value: str | os.PathLike[str]) -> Path:
        path = Path(raw_value or default_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def build_agent_adapter() -> AgentAdapter:
    return ProjectAgentAdapter()
