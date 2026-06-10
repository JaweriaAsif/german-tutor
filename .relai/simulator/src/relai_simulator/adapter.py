from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from german_tutor.cli import run_turn as run_graph_turn
from german_tutor.graph import (
    DEFAULT_MODEL,
    OFFTOPIC_REPLY,
    ROUTER_SYSTEM,
    SPECIALIST_PROMPTS,
    TutorState,
    _RouteDecision,
)
from german_tutor.persistence import Store
from german_tutor.srs import SrsState, review
from german_tutor.tools_lc import make_tools
from relai_simulator.adapter_contract import AgentAdapter, AgentTurnResult, ToolCallRecord, ToolResultRecord


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SIMULATOR_ROOT = PROJECT_ROOT / ".relai" / "simulator"
DEFAULT_RUNTIME_DIR = SIMULATOR_ROOT / "runtime"


@dataclass(slots=True)
class RuntimeConfig:
    learner_id: str
    db_path: Path
    checkpoint_db_path: Path


def _project_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


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
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        text = "".join(parts).strip()
        return text or None
    return str(content)


def _runtime_config(*, isolated: bool) -> RuntimeConfig:
    load_dotenv()
    DEFAULT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    session_id = uuid.uuid4().hex
    learner_id = os.getenv("TUTOR_LEARNER_ID") or f"relai-sim-{session_id[:8]}"

    if isolated:
        db_path = Path(os.getenv("TUTOR_DB", DEFAULT_RUNTIME_DIR / f"{session_id}-progress.db"))
        checkpoint_db_path = Path(
            os.getenv(
                "RELAI_SIMULATOR_CHECKPOINT_DB",
                DEFAULT_RUNTIME_DIR / f"{session_id}-graph.db",
            )
        )
    else:
        db_path = Path(os.getenv("TUTOR_DB", DEFAULT_RUNTIME_DIR / "component-progress.db"))
        checkpoint_db_path = Path(
            os.getenv(
                "RELAI_SIMULATOR_CHECKPOINT_DB",
                DEFAULT_RUNTIME_DIR / "component-graph.db",
            )
        )

    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()
    if not checkpoint_db_path.is_absolute():
        checkpoint_db_path = (PROJECT_ROOT / checkpoint_db_path).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig(
        learner_id=learner_id,
        db_path=db_path,
        checkpoint_db_path=checkpoint_db_path,
    )


def _build_simulator_graph(store: Store, learner_id: str, tool_groups: dict[str, list[BaseTool]]):
    llm = ChatOpenAI(model=_project_model(), temperature=0.3)
    router_llm = ChatOpenAI(model=_project_model(), temperature=0).with_structured_output(_RouteDecision)

    specialists = {
        name: create_react_agent(llm, tool_groups[name], prompt=prompt)
        for name, prompt in SPECIALIST_PROMPTS.items()
    }

    def router(state: TutorState) -> dict[str, object]:
        messages = state["messages"]
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        summary = store.welcome_back(learner_id)
        decision = router_llm.invoke(
            [
                SystemMessage(ROUTER_SYSTEM.format(state_summary=summary)),
                HumanMessage(text),
            ]
        )
        return {"route": decision.route}

    def make_specialist_node(name: str):
        agent = specialists[name]

        def node(state: TutorState) -> dict[str, object]:
            messages = state["messages"]
            result = agent.invoke({"messages": messages})
            new_messages = result["messages"][len(messages):]
            return {"messages": new_messages}

        return node

    def offtopic_node(state: TutorState) -> dict[str, object]:
        del state
        return {"messages": [AIMessage(content=OFFTOPIC_REPLY)]}

    workflow = StateGraph(TutorState)
    workflow.add_node("router", router)
    for name in SPECIALIST_PROMPTS:
        workflow.add_node(name, make_specialist_node(name))
    workflow.add_node("offtopic", offtopic_node)
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {name: name for name in (*SPECIALIST_PROMPTS, "offtopic")},
    )
    for name in (*SPECIALIST_PROMPTS, "offtopic"):
        workflow.add_edge(name, END)
    return workflow


def _flatten_tool_groups(tool_groups: dict[str, list[BaseTool]]) -> list[BaseTool]:
    tools_by_name: dict[str, BaseTool] = {}
    for tool_list in tool_groups.values():
        for tool in tool_list:
            tools_by_name.setdefault(tool.name, tool)
    return list(tools_by_name.values())


def _tool_by_name(tool_name: str, config: RuntimeConfig) -> BaseTool:
    store = Store(config.db_path)
    tools = _flatten_tool_groups(make_tools(store, config.learner_id))
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise KeyError(f"Unknown tutor tool: {tool_name}")


def component_set_level(level: str) -> str:
    return _tool_by_name("set_level", _runtime_config(isolated=False)).invoke({"level": level})


def component_get_unit_details(unit_id: str) -> str:
    return _tool_by_name("get_unit_details", _runtime_config(isolated=False)).invoke({"unit_id": unit_id})


def component_review_srs(
    ease: float = 2.5,
    interval: int = 0,
    reps: int = 0,
    quality: int = 5,
) -> dict[str, float | int]:
    state = review(SrsState(ease=ease, interval=interval, reps=reps), quality)
    return {"ease": state.ease, "interval": state.interval, "reps": state.reps}


class ProjectAgentAdapter:
    def __init__(self) -> None:
        config = _runtime_config(isolated=True)
        self._runtime = config
        self._store = Store(config.db_path)
        self._store.get_or_create_learner(config.learner_id)
        self._tool_groups = make_tools(self._store, config.learner_id)
        self.agent_or_tools = _flatten_tool_groups(self._tool_groups)
        self._checkpointer_context = SqliteSaver.from_conn_string(str(config.checkpoint_db_path))
        self._checkpointer = self._checkpointer_context.__enter__()
        self._graph = _build_simulator_graph(
            self._store,
            config.learner_id,
            self._tool_groups,
        ).compile(checkpointer=self._checkpointer)
        self._graph_config = {
            "configurable": {
                "thread_id": f"learner-{config.learner_id}-{uuid.uuid4().hex[:8]}",
            }
        }
        self._message_count = 0

    async def run_turn(self, user_message: str) -> AgentTurnResult:
        result = await asyncio.to_thread(
            run_graph_turn,
            self._graph,
            user_message,
            self._graph_config,
        )
        state = await asyncio.to_thread(self._graph.get_state, self._graph_config)
        all_messages = list(state.values.get("messages", []))
        new_messages = all_messages[self._message_count:]
        self._message_count = len(all_messages)

        tool_calls, tool_results = _collect_tool_records(new_messages)
        assistant_message = result if isinstance(result, str) else _last_ai_message(new_messages)
        return AgentTurnResult(
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    def close(self) -> None:
        context = getattr(self, "_checkpointer_context", None)
        if context is not None:
            self._checkpointer_context = None
            context.__exit__(None, None, None)

    def __del__(self) -> None:
        self.close()


def _last_ai_message(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = _stringify_content(message.content)
            if text:
                return text
    return None


def _collect_tool_records(messages: list[Any]) -> tuple[list[ToolCallRecord], list[ToolResultRecord]]:
    tool_calls: list[ToolCallRecord] = []
    tool_results: list[ToolResultRecord] = []
    tool_names_by_call_id: dict[str, str] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                call_id = tool_call.get("id")
                name = str(tool_call.get("name", "tool"))
                if call_id:
                    tool_names_by_call_id[str(call_id)] = name
                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        arguments=tool_call.get("args", {}),
                        call_id=str(call_id) if call_id else None,
                    )
                )
        elif isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", None)
            name = (
                getattr(message, "name", None)
                or tool_names_by_call_id.get(str(call_id))
                or "tool"
            )
            status = getattr(message, "status", None)
            content = _stringify_content(message.content)
            tool_results.append(
                ToolResultRecord(
                    name=str(name),
                    result=None if status == "error" else content,
                    error=content if status == "error" else None,
                    call_id=str(call_id) if call_id else None,
                )
            )

    return tool_calls, tool_results


def build_agent_adapter() -> AgentAdapter:
    return ProjectAgentAdapter()
