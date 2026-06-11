"""LangGraph orchestration for the German tutor.

Topology:

    START -> router --(conditional)--> {placement, lesson, grammar, vocab,
                                         exercise, conversation, progress,
                                         concierge, offtopic} -> END

The router classifies each learner turn (also acting as the on-topic guardrail)
and the chosen specialist node — a prebuilt ReAct agent with its own tools —
produces the reply. Conversation/graph state (incl. message history) is persisted
by the checkpointer the caller passes to `.compile(...)`, keyed by thread_id =
learner id, which is what makes "resume" work across runs. Durable learning
progress lives in the SQLite `Store` reached through the tools.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .persistence import Store
from .tools_lc import make_tools

DEFAULT_MODEL = "gpt-5.4"

LEVEL_RULE = (
    "Always teach at the learner's current CEFR level. Never use grammar or "
    "vocabulary clearly above their level without flagging it as a preview. "
    "Explanations are in English (with German examples); German you ask the learner "
    "to read/produce must match their level. Never invent grammar rules; if unsure, "
    "say so. Be encouraging and concise."
)

TTS_HINT = (
    " Pronunciation audio: call speak_german(text) AT MOST ONCE per turn, and only "
    "for the single new word or one short example the learner is on right now. Do "
    "NOT pre-generate audio for the whole lesson or call it several times in one "
    "turn — pace it one item at a time, in step with the conversation. If the "
    "learner wants to hear something specific, voice just that."
)

MOMENTUM = (
    " Keep momentum. When the learner answers, or says 'continue' / 'yes' / 'next' / "
    "'weiter' / 'ok', move straight into the next step yourself. Do NOT stop to ask "
    "'do you want to continue?' and do NOT re-list the activity menu every turn. End "
    "a turn only with the single concrete thing you need from the learner now (e.g. "
    "the answer to one exercise) — not a menu of options. Offer other activities "
    "(vocab, quiz, conversation) only when the lesson is actually finished or the "
    "learner asks what else they can do."
)

GROUND_HINT = (
    " Ground your teaching in real material: call get_lesson_material(topic) to pull "
    "the live Wikibooks German-course text for the current grammar point/topic, and "
    "base your explanation and examples on it (adapted to the learner's level). Do "
    "not invent grammar rules; if the material doesn't cover something, say so."
)

Route = Literal[
    "placement", "lesson", "grammar", "vocab", "exercise",
    "conversation", "progress", "concierge", "offtopic",
]

SPECIALIST_PROMPTS: dict[str, str] = {
    "placement": (
        "You run a short adaptive placement check. Ask 3-5 escalating questions "
        "(comprehension + a small production task) spanning A1 to B2, ONE at a time. "
        "Stop early once the level is clear, then call set_level with your estimate "
        "and explain why briefly. " + LEVEL_RULE
    ),
    "lesson": (
        "Deliver a paced lesson for the current unit, ONE step at a time. First call "
        "get_next_unit (or use the saved pointer) to know the unit, then "
        "get_cached_lesson(unit_id): if a cached lesson exists, RESUME from the saved "
        "pointer instead of regenerating. Otherwise design the lesson (intro, "
        "teaching, a checkpoint question, a few exercises, vocab, wrap-up), serialize "
        "it and call cache_lesson so future resumes are identical. After each step "
        "call save_lesson_pointer(unit_id, step). Add new words with add_vocab. Grade "
        "answers, record_attempt, and log_error on mistakes. Update mastery at the "
        "end. Never present the whole lesson at once. When get_learner_state shows "
        "is_absolute_beginner=true, or the pointer is empty / at step 0, treat this "
        "as the learner's very first lesson turn and teach for zero prior knowledge: "
        "explain every German word or phrase in plain English, give simple text "
        "pronunciation help (for example an English approximation plus stress note), "
        "include 2-3 tiny glossed examples, explicitly say what to notice, and make "
        "the learner task fully explicit before asking for a response. Do not rely "
        "on audio alone or on implicit classroom conventions; the learner should be "
        "able to continue even if they know no German yet. For those absolute-"
        "beginner / step-0 openings, every German item that appears in an example "
        "must either already have been taught in that same turn or be immediately "
        "glossed inline in English on the same line. Do not include preview example "
        "text with unexplained German, even if you say you will explain it later. "
        "If an example needs extra German words, gloss each new item immediately or "
        "replace the example with one that only uses already-explained German. Keep "
        "the step small but not skeletal. " + LEVEL_RULE + TTS_HINT + MOMENTUM + GROUND_HINT
    ),
    "grammar": (
        "Explain the requested grammar point for the learner's level: a short rule, "
        "2-3 example sentences with English glosses, and one common pitfall. Offer to "
        "drill it afterwards. " + LEVEL_RULE + TTS_HINT + GROUND_HINT
    ),
    "vocab": (
        "Teach or review vocabulary. For new words use add_vocab (include the article "
        "for nouns, e.g. 'der Tisch'). For review, call get_due_vocab, quiz ONE card "
        "at a time, and after each answer call review_vocab with a quality 0-5 (5 = "
        "instant recall, <3 = failed). " + LEVEL_RULE + TTS_HINT + MOMENTUM
    ),
    "exercise": (
        "Generate level-appropriate exercises for the current unit. Decide each "
        "exercise's expected answer up front so grading is consistent. Present ONE "
        "exercise at a time, wait for the answer, grade it, call record_attempt and "
        "log_error as needed, give feedback, then continue. Never dump them all at "
        "once. " + LEVEL_RULE + MOMENTUM
    ),
    "conversation": (
        "Role-play a realistic German dialogue (café, directions, interview, ...) at "
        "the learner's level. Speak mostly in German, keep turns short, gently correct "
        "serious mistakes inline with a brief English note, and log_error on notable "
        "ones. " + LEVEL_RULE + TTS_HINT
    ),
    "progress": (
        "Summarize how the learner is doing in plain text, call update_mastery for the "
        "relevant unit (mark completed only when mastery >= 0.8), recommend advance / "
        "review / repeat with a one-line rationale, and call log_session_summary."
    ),
    "concierge": (
        "You are the friendly front desk of a German tutor. Greet the learner "
        "(reference where they left off if returning) and answer brief questions. "
        "Offer the activity menu (lesson, grammar, vocab review, quiz, conversation, "
        "placement, progress) ONLY on a first greeting or when the learner explicitly "
        "asks what they can do — do not repeat the menu after every turn. If the "
        "learner clearly wants to keep learning, point them straight into it rather "
        "than re-listing options. Keep it short. " + LEVEL_RULE
    ),
}

ROUTER_SYSTEM = (
    "You route a learner's message in a German-tutoring app to the right specialist. "
    "Learner state: {state_summary}\n\n"
    "Choose exactly one route:\n"
    "- placement: unknown level, or learner wants to be placed/tested for level\n"
    "- lesson: 'lesson', 'teach me', 'continue', 'resume', a topic/unit, OR a bare "
    "affirmation/continuation ('yes', 'ok', 'next', 'weiter', 'go on', 'sure') when "
    "a lesson is in progress\n"
    "- grammar: a grammar question or 'explain ...'\n"
    "- vocab: 'vocab', 'words', 'flashcards', 'review words'\n"
    "- exercise: 'quiz', 'exercises', 'drill', 'practice questions'\n"
    "- conversation: 'talk', 'conversation', 'role-play', 'sprechen'\n"
    "- progress: 'progress', 'how am I doing'\n"
    "- concierge: greetings, menu, vague/short messages, or anything that needs a quick reply\n"
    "- offtopic: NOT about learning German (e.g. write code, unrelated trivia, "
    "legal/medical/financial advice) -> these must be refused\n"
)

OFFTOPIC_REPLY = (
    "Das ist leider außerhalb meines Themas. I'm your German tutor, so let's keep it "
    "to learning German. Try /lesson, /review, or ask me a grammar question!"
)


class _RouteDecision(BaseModel):
    route: Route = Field(description="The single best route for this message.")


class TutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str


def _model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


AUDIO_MARKUP_HINT = (
    " IMPORTANT for pronunciation: wrap EVERY standalone German word, phrase, or "
    "example sentence the learner sees in double square brackets, like [[de:der "
    "Tisch]] or [[de:Ich heiße Anna.]]. The interface turns these into clickable "
    "audio buttons. Wrap only real German (never English), and do it for the key "
    "words and examples — not every little function word."
)

_AUDIO_AGENTS = ("placement", "lesson", "grammar", "vocab", "exercise", "conversation")


def build_tutor_graph(store: Store, learner_id: str, audio_markup: bool = False):
    """Build (uncompiled) the tutor StateGraph for one learner.

    Compile it with a checkpointer in the caller:
        graph = build_tutor_graph(store, lid).compile(checkpointer=saver)

    When audio_markup is True, teaching agents wrap German text in [[de:...]] markers
    so the CLI can render clickable pronunciation buttons. (Off by default so RELAI
    simulations see clean transcripts.)
    """
    llm = ChatOpenAI(model=_model(), temperature=0.3)
    router_llm = ChatOpenAI(model=_model(), temperature=0).with_structured_output(_RouteDecision)
    tools = make_tools(store, learner_id)

    def _prompt(name: str, base: str) -> str:
        if audio_markup and name in _AUDIO_AGENTS:
            return base + AUDIO_MARKUP_HINT
        return base

    # One ReAct agent per specialist, each with only the tools it needs.
    specialists = {
        name: create_react_agent(llm, tools[name], prompt=_prompt(name, prompt))
        for name, prompt in SPECIALIST_PROMPTS.items()
    }

    def router(state: TutorState) -> dict:
        last = state["messages"][-1]
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

        def node(state: TutorState) -> dict:
            # Raise the step budget: a lesson turn can legitimately make many tool
            # calls (state, curriculum, material, cache, pointer, vocab, audio), and
            # the default recursion_limit (25) can cut a turn off before its final
            # reply — which surfaces to the user as "no answer".
            result = agent.invoke(
                {"messages": state["messages"]}, config={"recursion_limit": 60}
            )
            # Return only the messages this sub-agent newly produced.
            new = result["messages"][len(state["messages"]):]
            return {"messages": new}

        return node

    def offtopic_node(state: TutorState) -> dict:
        return {"messages": [AIMessage(content=OFFTOPIC_REPLY)]}

    workflow = StateGraph(TutorState)
    workflow.add_node("router", router)
    for name in SPECIALIST_PROMPTS:
        workflow.add_node(name, make_specialist_node(name))
    workflow.add_node("offtopic", offtopic_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {name: name for name in (*SPECIALIST_PROMPTS, "offtopic")},
    )
    for name in (*SPECIALIST_PROMPTS, "offtopic"):
        workflow.add_edge(name, END)

    return workflow
