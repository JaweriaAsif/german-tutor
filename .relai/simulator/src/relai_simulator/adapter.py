from __future__ import annotations

import asyncio
import atexit
import json
import os
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver

import german_tutor.graph as graph_module
from german_tutor.persistence import DEFAULT_DB_PATH, Store
from german_tutor.tools_lc import make_tools
from relai_simulator.adapter_contract import (
    AgentAdapter,
    AgentTurnResult,
    ToolCallRecord,
    ToolResultRecord,
)

DEFAULT_CHECKPOINT_DB = Path(".german_tutor/graph.db")


class ProjectAgentAdapter:
    def __init__(self) -> None:
        self.project_root = _project_root()
        self.learner_id = os.getenv("TUTOR_LEARNER_ID", "relai")
        self.thread_id = os.getenv("TUTOR_THREAD_ID", f"relai-{uuid.uuid4().hex}")
        self.store = Store(_runtime_db_path("TUTOR_DB", DEFAULT_DB_PATH, self.project_root))
        self.store.get_or_create_learner(self.learner_id)
        self._checkpointer_cm = SqliteSaver.from_conn_string(
            str(_runtime_db_path("TUTOR_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB, self.project_root))
        )
        self._checkpointer = self._checkpointer_cm.__enter__()
        self.graph, self.agent_or_tools = _compile_graph_with_captured_tools(
            store=self.store,
            learner_id=self.learner_id,
            checkpointer=self._checkpointer,
        )
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self._message_count = 0
        self._closed = False
        atexit.register(self.close)

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        result = await asyncio.to_thread(
            self.graph.invoke,
            {"messages": [HumanMessage(content=user_message)]},
            config=self.config,
        )
        messages = list(result.get("messages", []))
        new_messages = messages[self._message_count :]
        self._message_count = len(messages)

        tool_calls, tool_results = _extract_tool_events(new_messages)
        assistant_message = _last_assistant_message(new_messages) or "(no response)"
        return AgentTurnResult(
            assistant_message=assistant_message,
            metadata={
                "learner_id": self.learner_id,
                "thread_id": self.thread_id,
            },
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            self.store.close()
        with suppress(Exception):
            self._checkpointer_cm.__exit__(None, None, None)


def build_agent_adapter() -> AgentAdapter:
    return ProjectAgentAdapter()


def component_get_learner_state() -> str:
    return _invoke_component_tool("get_learner_state")


def component_set_level(level: str) -> str:
    return _invoke_component_tool("set_level", level=level)


def component_get_next_unit() -> str:
    return _invoke_component_tool("get_next_unit")


def component_get_unit_details(unit_id: str) -> str:
    return _invoke_component_tool("get_unit_details", unit_id=unit_id)


def component_save_lesson_pointer(unit_id: str, step: int) -> str:
    return _invoke_component_tool("save_lesson_pointer", unit_id=unit_id, step=step)


def component_cache_lesson(unit_id: str, lesson_json: str) -> str:
    return _invoke_component_tool("cache_lesson", unit_id=unit_id, lesson_json=lesson_json)


def component_get_cached_lesson(unit_id: str) -> str:
    return _invoke_component_tool("get_cached_lesson", unit_id=unit_id)


def component_record_attempt(
    unit_id: str,
    exercise_id: str,
    correct: bool,
    score: float,
) -> str:
    return _invoke_component_tool(
        "record_attempt",
        unit_id=unit_id,
        exercise_id=exercise_id,
        correct=correct,
        score=score,
    )


def component_log_error(category: str, example: str, correction: str) -> str:
    return _invoke_component_tool(
        "log_error",
        category=category,
        example=example,
        correction=correction,
    )


def component_update_mastery(unit_id: str, mastery: float, status: str) -> str:
    return _invoke_component_tool(
        "update_mastery",
        unit_id=unit_id,
        mastery=mastery,
        status=status,
    )


def component_add_vocab(lemma: str, gloss: str) -> str:
    return _invoke_component_tool("add_vocab", lemma=lemma, gloss=gloss)


def component_get_due_vocab() -> str:
    return _invoke_component_tool("get_due_vocab")


def component_review_vocab(card_id: int, quality: int) -> str:
    return _invoke_component_tool("review_vocab", card_id=card_id, quality=quality)


def component_log_session_summary(summary: str) -> str:
    return _invoke_component_tool("log_session_summary", summary=summary)


def _invoke_component_tool(tool_name: str, **kwargs: Any) -> str:
    store = Store(_runtime_db_path("TUTOR_DB", DEFAULT_DB_PATH, _project_root()))
    try:
        learner_id = os.getenv("TUTOR_LEARNER_ID", "relai")
        store.get_or_create_learner(learner_id)
        tool = _tool_lookup(store, learner_id)[tool_name]
        payload = kwargs if kwargs else {}
        return str(tool.invoke(payload))
    finally:
        store.close()


def _compile_graph_with_captured_tools(
    *,
    store: Store,
    learner_id: str,
    checkpointer: Any,
) -> tuple[Any, list[BaseTool]]:
    captured: dict[str, dict[str, list[BaseTool]]] = {}
    original_make_tools = graph_module.make_tools

    def capture_make_tools(bound_store: Store, bound_learner_id: str) -> dict[str, list[BaseTool]]:
        tools = original_make_tools(bound_store, bound_learner_id)
        captured["tools"] = tools
        return tools

    graph_module.make_tools = capture_make_tools
    try:
        graph = graph_module.build_tutor_graph(store, learner_id).compile(checkpointer=checkpointer)
    finally:
        graph_module.make_tools = original_make_tools

    return graph, _flatten_tools(captured["tools"])


def _extract_tool_events(
    messages: list[BaseMessage],
) -> tuple[list[ToolCallRecord], list[ToolResultRecord]]:
    tool_calls: list[ToolCallRecord] = []
    tool_results: list[ToolResultRecord] = []
    names_by_call_id: dict[str, str] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                call_id = _optional_string(tool_call.get("id"))
                name = _optional_string(tool_call.get("name")) or "unknown_tool"
                if call_id is not None:
                    names_by_call_id[call_id] = name
                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        arguments=tool_call.get("args", {}),
                        call_id=call_id,
                        metadata=_message_metadata(message),
                    )
                )
        elif isinstance(message, ToolMessage):
            call_id = _optional_string(getattr(message, "tool_call_id", None))
            tool_results.append(
                ToolResultRecord(
                    name=_optional_string(getattr(message, "name", None))
                    or (names_by_call_id.get(call_id) if call_id else None)
                    or "unknown_tool",
                    result=_message_content(message),
                    call_id=call_id,
                    metadata=_message_metadata(message),
                )
            )

    return tool_calls, tool_results


def _last_assistant_message(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _stringify_content(message.content)
            if text:
                return text
    return None


def _message_content(message: BaseMessage) -> object:
    content = getattr(message, "content", None)
    return _stringify_content(content) if isinstance(content, (str, list)) else content


def _message_metadata(message: BaseMessage) -> dict[str, object]:
    metadata: dict[str, object] = {
        "message_type": type(message).__name__,
    }
    name = getattr(message, "name", None)
    if name:
        metadata["message_name"] = str(name)
    message_id = getattr(message, "id", None)
    if message_id:
        metadata["message_id"] = str(message_id)
    return metadata


def _stringify_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                    continue
                if item.get("type") == "text" and item.get("content") is not None:
                    parts.append(str(item["content"]))
        if parts:
            return "\n".join(part for part in parts if part)
        return json.dumps(content)
    return str(content)


def _flatten_tools(tool_groups: dict[str, list[BaseTool]]) -> list[BaseTool]:
    flattened: list[BaseTool] = []
    seen: set[str] = set()
    for tools in tool_groups.values():
        for tool in tools:
            name = getattr(tool, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            flattened.append(tool)
    return flattened


def _tool_lookup(store: Store, learner_id: str) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in _flatten_tools(make_tools(store, learner_id))}


def _runtime_db_path(env_name: str, default_relative_path: Path, project_root: Path) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return (project_root / default_relative_path).resolve()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
