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

import time, json, os
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import get_llm_model  # Gemini 2.5 Flash model

LOG_PATH = os.environ.get("REWRITE_LOG", "logs/model_selection_log.txt")


def _simple_rewrite(query: str) -> str:
    """Fallback rule-based rewrite (no model)."""
    q = query.strip()
    if not q.endswith("?"):
        q += "?"
    if len(q.split()) < 4:
        q = f"Please provide a detailed, step-by-step answer for: {q}"
    return q


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

        # Modern LangChain syntax
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


def run(state: Dict) -> Dict:
    """Main node entry point for rewriting user query."""
    query = state.get("question") or state.get("query") or ""

    print(f"[TRACE] Entered rewrite_query | Original Query: {query!r}")

    if not query:
        print("[rewrite_query] ⚠️ No query found in state — skipping rewrite.")
        state.update({
            "rewritten_query": "",
            "prompt": "",
            "question": "",
            "rewrite_metadata": {"method": "noop", "ts": time.time()}
        })
        return state

    # === 1️⃣ Try Gemini LLM for rewriting ===
    rewritten = _rewrite_with_gemini(query)
    method = "Gemini-2.5-Flash"

    # === 2️⃣ Update state ===
    state.update({
        "rewritten_query": rewritten,
        "prompt": rewritten,
        "question": rewritten,  # required by retrieve.py
        "rewrite_metadata": {"method": method, "ts": time.time()}
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

    # === 4️⃣ Always print rewritten query clearly ===
    print(f"[rewrite_query] ✅ Original: {query!r}")
    print(f"[rewrite_query] ✨ Rewritten: {rewritten!r}\n")

    return state
