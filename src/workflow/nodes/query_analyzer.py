# src/workflow/nodes/query_analyzer.py
"""
Query Analyzer Node
-------------------
Analyzes the rewritten query for completeness and clarity.
If the question is vague or missing context, sets needs_more_detail=True.

Inputs
------
state["question"]: str   # rewritten query from Query Rewriter

Outputs
-------
state["analysis_result"]: str  # explanation of decision
state["needs_more_detail"]: bool
"""

import time, json, os
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import get_llm_model

LOG_PATH = os.environ.get("ANALYZER_LOG", "logs/feedback_log.txt")

def _fallback_analysis(question: str) -> Dict[str, str]:
    """Fallback heuristic if LLM fails."""
    vague_terms = {"explain", "describe", "tell", "information", "details", "overview"}
    is_vague = any(word.lower() in question.lower() for word in vague_terms)
    reason = "Contains generic or vague keywords." if is_vague else "Appears specific enough."
    return {"needs_more_detail": is_vague, "analysis_result": reason}


def _analyze_with_llm(question: str) -> Dict[str, str]:
    """
    Uses Gemini 2.0 Flash (LangChain) to evaluate clarity of query.
    Returns structured decision.
    """
    try:
        llm = get_llm_model()

        prompt_text = (
            "You are a Query Analysis Agent.\n"
            "Decide if the following user query has enough specific details "
            "for an AI retrieval system to find accurate information.\n\n"
            "Output JSON:\n"
            "{\n"
            "  \"needs_more_detail\": true/false,\n"
            "  \"reason\": \"<brief reasoning>\"\n"
            "}\n\n"
            "User Query:\n{question}"
        )

        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question}).strip()

        # Parse simple JSON-ish text from model
        import re, json
        json_text = re.search(r"\{.*\}", response, re.DOTALL)
        if json_text:
            data = json.loads(json_text.group(0))
            return {
                "needs_more_detail": bool(data.get("needs_more_detail", False)),
                "analysis_result": data.get("reason", "").strip()
            }

        # fallback if no JSON
        return _fallback_analysis(question)

    except Exception as e:
        print(f"[query_analyzer] Fallback due to error: {e}")
        return _fallback_analysis(question)


def run(state: Dict) -> Dict:
    """Main entry for analyzer node."""
    question = state.get("question") or state.get("rewritten_query") or ""
    if not question:
        state["needs_more_detail"] = False
        state["analysis_result"] = "No query provided."
        return state

    result = _analyze_with_llm(question)

    state["needs_more_detail"] = result["needs_more_detail"]
    state["analysis_result"] = result["analysis_result"]
    state["analysis_metadata"] = {
        "method": "Gemini-2.0-Flash",
        "timestamp": time.time(),
    }

    # Log result
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "question": question,
                "needs_more_detail": result["needs_more_detail"],
                "reason": result["analysis_result"]
            }) + "\n")
    except Exception:
        pass

    print(f"[query_analyzer] ✅ Query analysis: needs_more_detail={result['needs_more_detail']} ({result['analysis_result']})")
    return state
