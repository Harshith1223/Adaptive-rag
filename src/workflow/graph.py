# src/workflow/graph.py
from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
import datetime
import re

from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.router import run as router_run  # ✅ new adaptive router
from src.workflow.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_documents import grade_documents
from src.workflow.nodes.retrieve import retrieve as retrieve_run
from src.workflow.nodes.web_search import web_search
from src.workflow.state import GraphState
from src.workflow.nodes.rewrite_query import run as rewrite_query_run
from src.workflow.nodes.query_analyzer import run as query_analyzer_run

load_dotenv()
MAX_RETRIES = 3
checkpointer = MemorySaver()

# ---------------------------------------------------------------------------- #
# 🧠 Utility Functions
# ---------------------------------------------------------------------------- #

def log(msg: str):
    """Timestamped print for better tracing."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")


def decide_to_generate(state):
    log(f"---ASSESS DOCUMENTS: web_search={state.get('web_search', False)}---")
    return WEBSEARCH if state.get("web_search", False) else GENERATE


def is_greeting(query: str) -> bool:
    """Strict greeting detection using regex word boundaries."""
    if not isinstance(query, str):
        return False

    query = query.lower().strip()
    greetings = [
        "hi", "hello", "hey", "greetings",
        "good morning", "good afternoon", "good evening",
        "howdy", "yo", "sup"
    ]

    pattern = r"\b(" + "|".join(re.escape(g) for g in greetings) + r")\b"
    return re.search(pattern, query) is not None


def grade_generation_grounded_in_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state.get("question", "").lower().strip()
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retry_count = state.get("retry_count", 0)

    print(f"---GENERATED ANSWER: {generation}---")

    # ✅ 1. Skip grading only for actual greetings (not substring)
    if is_greeting(question) or is_greeting(generation):
        print("---SKIPPING GRADING: Greeting detected---")
        return "useful"

    # ✅ 2. Handle empty or failed generations
    if not generation.strip():
        print("---EMPTY GENERATION: Retrying---")
        state["retry_count"] = retry_count + 1
        return "not supported"

    # ✅ 3. Hallucination grading
    try:
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        if not score or not hasattr(score, "binary_score"):
            print("---HALLUCINATION GRADER FAILED: Treating as useful---")
            return "useful"

        print(f"---HALLUCINATION SCORE: {score.binary_score}---")

        # ✅ 4. Retry logic
        if retry_count >= MAX_RETRIES:
            print("---MAX RETRIES HIT: ACCEPT AS-IS---")
            return "useful"

        if score.binary_score:
            answer_score = answer_grader.invoke({"question": question, "generation": generation})
            print(f"---ANSWER SCORE: {answer_score.binary_score}---")
            return "useful" if answer_score.binary_score else "not useful"
        else:
            state["retry_count"] = retry_count + 1
            print(f"---HALLUCINATION FAILED: RETRY {retry_count + 1}/{MAX_RETRIES}---")
            return "not supported"

    except Exception as e:
        print(f"⚠️ [grade_generation] Error: {e}")
        return "useful"

# ---------------------------------------------------------------------------- #
# 🧩 Entry Point Logic
# ---------------------------------------------------------------------------- #

def route_question(state: GraphState) -> str:
    """
    Entry logic before workflow branching.
    Ensures 'question' field exists and initializes pipeline correctly.
    """
    log("---ROUTE QUESTION (entry)---")

    # Normalize the input
    question = (
        state.get("question")
        or state.get("user_query")
        or ""
    ).strip()

    # ✅ Always set 'question' in the state
    state["question"] = question

    # Debug info
    log(f"[DEBUG] question='{question}' | keys={list(state.keys())}")

    # Handle empty input
    if not question:
        log("---No question provided, routing to GENERATE---")
        state["generation"] = "Please provide a valid question to begin analysis."
        return GENERATE

    # Handle greetings
    if is_greeting(question):
        print("---ROUTED TO: GREETING (direct generation)---")
        state["question"] = question or "Hello"
        state["generation"] = (
            "Hello! 👋 How can I assist you today? I can help with AI, projects, or general questions."
        )
        state["web_search"] = False
        state["documents"] = []
        state["retry_count"] = 0
        return GENERATE



    # ✅ Normal route
    log("---Routing to Rewriter---")
    return "rewrite_query"

# ---------------------------------------------------------------------------- #
# ⚙️ Workflow Definition
# ---------------------------------------------------------------------------- #

workflow = StateGraph(GraphState)

# === Nodes ===
workflow.add_node("rewrite_query", RunnableLambda(rewrite_query_run))
workflow.add_node("query_analyzer", RunnableLambda(query_analyzer_run))
workflow.add_node("router", RunnableLambda(router_run))
workflow.add_node("retrieve", RunnableLambda(retrieve_run))
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# === Connections ===
workflow.add_edge("rewrite_query", "query_analyzer")
workflow.add_edge("query_analyzer", "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state.get("selected_source"),
    {
        "vector": RETRIEVE,
        "web": WEBSEARCH,
        "tool": RETRIEVE,  # Placeholder for MCP tools
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)

# === Entry point ===
workflow.set_conditional_entry_point(
    route_question,
    {"rewrite_query": "rewrite_query", GENERATE: GENERATE},
)

# ✅ Compile graph
app = workflow.compile(checkpointer=checkpointer)
