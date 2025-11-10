"""
Query Rewriter Node
-------------------
Uses Gemini 2.5 Flash (via LangChain) to rewrite the user query
into a clear, retrieval-friendly version.

Inputs:
    state["question"]  : str

Outputs:
    state["rewritten_query"] : str
    state["prompt"]          : str
    state["question"]        : str   # for retrieve.py compatibility
    state["rewrite_metadata"] : dict
"""

import time, json, os, re
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import get_llm_model  # Gemini 2.5 Flash model

LOG_PATH = os.environ.get("REWRITE_LOG", "logs/model_selection_log.txt")


# ------------------------------------------------------------------------- #
# 🧠 Helper Functions
# ------------------------------------------------------------------------- #

def _simple_rewrite(query: str) -> str:
    """Fallback rule-based rewrite (no model)."""
    q = query.strip()
    if not q.endswith("?"):
        q += "?"
    if len(q.split()) < 4:
        q = f"Please provide a detailed, step-by-step answer for: {q}"
    return q


def _sanitize_rewrite(original: str, candidate: str) -> str:
    """Clean up the rewritten query from Gemini or fallback output."""
    if not candidate or not isinstance(candidate, str):
        return original.strip()

    text = candidate.strip()

    # Remove excessive punctuation or artifacts
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"(\.\?|\?\.|[\.]{2,}|[\?]{2,})", ".", text)  # fix '.?' or '??'
    text = text.replace('"', '').replace("'", "")

    # Normalize trailing punctuation to a single question mark if it seems like a question
    if text.endswith(".") and "?" in original:
        text = text[:-1] + "?"
    text = re.sub(r"[\.?]+$", "?", text)

    # Ensure it's not empty or identical placeholder
    if not text or text.lower().strip() in {"?", ".", ""}:
        text = original.strip()

    return text.strip()


def _rewrite_with_gemini(query: str) -> str:
    """
    Calls Gemini 2.5 Flash to rewrite the user query.
    Returns the rewritten text only.
    """
    try:
        llm = get_llm_model()

        prompt_text = (
            "You are a Query Rewriter Agent.\n"
            "Rewrite the following user question to be self-contained, clear, and optimized "
            "for document retrieval. Keep the meaning the same but make it specific. "
            "Return only the rewritten query without extra commentary.\n\n"
            "User Query:\n{query}"
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        parser = StrOutputParser()
        chain = prompt | llm | parser

        rewritten = chain.invoke({"query": query})

        if isinstance(rewritten, dict):
            rewritten_text = rewritten.get("text", "").strip()
        else:
            rewritten_text = str(rewritten).strip()

        if not rewritten_text:
            print(f"[rewrite_query] ⚠️ Empty output from Gemini — using fallback.")
            rewritten_text = _simple_rewrite(query)

        return rewritten_text

    except Exception as e:
        print(f"[rewrite_query] ⚠️ Fallback due to error: {e}")
        return _simple_rewrite(query)


# ------------------------------------------------------------------------- #
# 🚀 Main Entry Function
# ------------------------------------------------------------------------- #

def run(state: Dict) -> Dict:
    """Main node entry point for rewriting user query."""
    query = state.get("question") or state.get("query") or ""

    print(f"[TRACE] Entered rewrite_query | Original Query: {query!r}")

    if not query.strip():
        print("[rewrite_query] ⚠️ No query found in state — skipping rewrite.")
        state.update({
            "rewritten_query": "",
            "prompt": "",
            "question": "",
            "rewrite_metadata": {"method": "noop", "ts": time.time(), "rewritten_query": ""}
        })
        return state

    # === 1️⃣ Try Gemini rewrite ===
    rewritten_raw = _rewrite_with_gemini(query)
    rewritten = _sanitize_rewrite(query, rewritten_raw)
    method = "Gemini-2.5-Flash"

    # === 2️⃣ Update state ===
    # Persist rewritten query both as a top-level key and inside metadata so LangGraph final state contains it
    state.update({
        "rewritten_query": rewritten,
        "prompt": rewritten,
        "question": rewritten,  # for retrieve.py compatibility
        "rewrite_metadata": {
            "method": method,
            "ts": time.time(),
            "rewritten_query": rewritten
        }
    })

    # === 3️⃣ Log rewrite event ===
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "original_query": query,
                "rewritten_query": rewritten,
                "method": method
            }) + "\n")
    except Exception as e:
        print(f"[rewrite_query] ⚠️ Failed to write log: {e}")

    # === 4️⃣ Console trace ===
    print(f"[rewrite_query] ✅ Original: {query!r}")
    print(f"[rewrite_query] ✨ Clean Rewritten: {rewritten!r}\n")

    return state
