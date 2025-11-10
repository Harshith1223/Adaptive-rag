# src/workflow/graph.py
from typing import Any, Dict
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
import datetime
import re
from src.workflow.state import GraphState
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Load summarizer once globally (for reference trimming)
from functools import lru_cache

# --- Internal Imports ---
from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.router import run as router_run
from src.workflow.consts import (
    RETRIEVE,
    GRADE_DOCUMENTS,
    GENERATE,
    WEBSEARCH,
    REWRITE_QUERY,
    QUERY_ANALYZER,
    ROUTER,
)
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_documents import grade_documents
from src.workflow.nodes.retrieve import run as retrieve_run
from src.workflow.nodes.web_search import web_search
from src.workflow.nodes.rewrite_query import run as rewrite_query_run
from src.workflow.nodes.query_analyzer import run as query_analyzer_run

# ---------------------------------------------------------------------------- #
# 🧠 Setup
# ---------------------------------------------------------------------------- #

load_dotenv()
MAX_RETRIES = 3
checkpointer = MemorySaver()

# ---------------------------------------------------------------------------- #
# 🧩 Utility Functions
# ---------------------------------------------------------------------------- #

def log(msg: str):
    """Timestamped print for better tracing."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")


def decide_to_generate(state):
    """Decides next step based on web_search flag."""
    log(f"---ASSESS DOCUMENTS: web_search={state.get('web_search', False)}---")
    return WEBSEARCH if state.get("web_search", False) else GENERATE


import re

def is_greeting(query: str) -> bool:
    """Detects if a user message is a greeting or salutation using pattern-based matching."""
    if not isinstance(query, str) or not query.strip():
        return False

    query = query.lower().strip()

    # Common greeting stems and variants
    greeting_patterns = [
        r"\bhi+\b",                          # hi, hii, hiiii
        r"\bhey+\b",                         # hey, heyy, heyyy
        r"\bhello+\b",                       # hello, hellooo
        r"\bhowdy\b",
        r"\b(good\s)?(morning|afternoon|evening|night)\b",
        r"\bsup\b", r"\bwhats? ?up\b",
        r"\byo+\b",
        r"\bgreetings?\b",
        r"👋", r"🙋", r"🙏"                   # emoji-based greetings
    ]

    # Match any of the greeting patterns
    pattern = "|".join(greeting_patterns)
    if re.search(pattern, query):
        return True

    # Fallback heuristic: short informal greetings with only 1–3 words
    if len(query.split()) <= 3 and any(word in query for word in ["hi", "hey", "hello", "yo", "sup", "morning"]):
        return True

    return False


# ---------------------------------------------------------------------------- #
# 🧩 Grading & Evaluation Function
# ---------------------------------------------------------------------------- #

@lru_cache()
def get_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"[metric] ⚠️ Summarizer load failed: {e}")
        return None

def grade_generation_grounded_in_documents_and_question(state):
    """Evaluates RAG answer using retrieval + generation focused metrics (with capped retries)."""
    print("---CHECK RAG PERFORMANCE (Retrieval + Generation metrics)---")

    question = state.get("question", "").strip().lower()
    documents = state.get("documents", [])
    generation = state.get("generation", "").strip()
    retry_count = state.get("retry_count", 0)

    # If no generation at all → retry
    if not generation:
        print("---EMPTY GENERATION: RETRYING---")
        state["retry_count"] = retry_count + 1
        return "not supported"

    try:
        # ==========================================================
        # 🧩 1. Hallucination Check (Factual Consistency)
        # ==========================================================
        hallucination = hallucination_grader.invoke({"documents": documents, "generation": generation})
        factual_ok = bool(getattr(hallucination, "binary_score", True))
        state["factual_consistency"] = 1.0 if factual_ok else 0.0
        print(f"---FACTUAL CONSISTENCY: {factual_ok}---")

        # ==========================================================
        # 🧠 2. Answer Relevance (Generation Relevance)
        # ==========================================================
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        relevance_ok = bool(getattr(answer_score, "binary_score", True))
        state["answer_relevance"] = 1.0 if relevance_ok else 0.0
        print(f"---ANSWER RELEVANCE: {relevance_ok}---")

        # ==========================================================
        # 📚 3. Retrieval Focused Metrics
        # ==========================================================
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        q_emb = embedder.embed_query(question)
        d_embs = [embedder.embed_query(doc.page_content) for doc in documents]

        sims = [np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)) for d_emb in d_embs]
        context_relevance = float(np.mean(sims)) if sims else 0.0
        state["context_relevance"] = round(context_relevance, 3)
        print(f"---CONTEXT RELEVANCE SCORE: {context_relevance:.3f}---")

        relevant_docs = sum(1 for s in sims if s > 0.6)  # slightly lowered threshold
        total_docs = len(sims)
        context_precision = round(relevant_docs / total_docs, 3) if total_docs else 0.0
        state["context_precision"] = context_precision
        print(f"---CONTEXT PRECISION: {context_precision:.3f}---")

        # ==========================================================
        # 🧮 4. Weighted Confidence Calculation
        # ==========================================================
        confidence = round((
            0.4 * state["answer_relevance"] +
            0.3 * state["factual_consistency"] +
            0.2 * state["context_relevance"] +
            0.1 * state["context_precision"]
        ), 3)

        state["confidence"] = confidence
        print(f"---COMBINED CONFIDENCE: {confidence:.3f}---")

        # ==========================================================
        # 🔁 5. Retry Logic with Lower Bar & Cap
        # ==========================================================
        CONFIDENCE_THRESHOLD = 0.35  # lowered threshold
        MAX_ALLOWED_RETRIES = 3      # absolute hard cap

        if not factual_ok or confidence < CONFIDENCE_THRESHOLD:
            if retry_count + 1 >= MAX_ALLOWED_RETRIES:
                print(f"---MAX RETRIES REACHED ({MAX_ALLOWED_RETRIES}) → ACCEPTING LAST ANSWER---")
                return "useful"
            print(f"---LOW CONFIDENCE ({confidence:.3f}) → RETRY {retry_count + 1}/{MAX_ALLOWED_RETRIES}---")
            state["retry_count"] = retry_count + 1
            return "not supported"

        # ==========================================================
        # ✅ Final Decision
        # ==========================================================
        return "useful" if relevance_ok else "not useful"

    except Exception as e:
        print(f"⚠️ [grade_generation] Error: {e}")
        return "useful"


# ---------------------------------------------------------------------------- #
# 🧭 Entry Point Logic
# ---------------------------------------------------------------------------- #

def route_question(state: GraphState) -> str:
    """Initial route setup for the workflow."""
    log("---ROUTE QUESTION (entry)---")

    # Normalize input
    question = (
        state.get("question")
        or state.get("question")
        or ""
    ).strip()

    # Ensure key presence
    state["question"] = question

    # Debug
    log(f"[DEBUG] question='{question}' | keys={list(state.keys())}")

    # Handle empty query
    if not question:
        log("---No question provided, routing to GENERATE---")
        state["generation"] = "Please provide a valid question to begin analysis."
        return GENERATE

    # Normal route
    log("---Routing to Rewriter---")
    return REWRITE_QUERY

# ---------------------------------------------------------------------------- #
# ⚙️ Workflow Definition
# ---------------------------------------------------------------------------- #

workflow = StateGraph(GraphState)

# === Nodes ===
workflow.add_node(REWRITE_QUERY, RunnableLambda(rewrite_query_run))
workflow.add_node(QUERY_ANALYZER, RunnableLambda(query_analyzer_run))
workflow.add_node(ROUTER, RunnableLambda(router_run))
workflow.add_node(RETRIEVE, RunnableLambda(retrieve_run))
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# === Edges ===
workflow.add_edge(REWRITE_QUERY, QUERY_ANALYZER)

workflow.add_conditional_edges(
    QUERY_ANALYZER,
    lambda state: REWRITE_QUERY if state.get("needs_more_detail", False) else ROUTER,
    {REWRITE_QUERY: REWRITE_QUERY, ROUTER: ROUTER},
)

workflow.add_conditional_edges(
    ROUTER,
    lambda state: state.get("selected_source"),
    {"vector": RETRIEVE, "web": WEBSEARCH, "tool": WEBSEARCH},
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
    {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
)

workflow.add_edge(WEBSEARCH, GENERATE)

# === Entry point ===
workflow.set_conditional_entry_point(
    route_question,
    {REWRITE_QUERY: REWRITE_QUERY, GENERATE: GENERATE},
)

# ✅ Compile workflow
app = workflow.compile(checkpointer=checkpointer)
