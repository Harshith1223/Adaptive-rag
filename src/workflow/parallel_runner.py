import asyncio
from typing import Dict, Any
from src.workflow.graph import app
from src.models.model import get_perplexity_llm, get_llm_model, get_openai_validator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, os
from datetime import datetime
from src.workflow.state import GraphState
from src.workflow.chains.hallucination_grader import hallucination_grader  # ✅ import your grader
from src.workflow.chains.answer_grader import answer_grader                # ✅ import second grader

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "model_selection_log.txt")


def build_cited_answer(answer: str, documents) -> str:
    """
    Append lightweight source citations to the selected final answer.
    """
    if not answer:
        return answer

    unique_sources = []
    seen = set()

    for doc in documents or []:
        source = (getattr(doc, "metadata", {}) or {}).get("source")
        if source and source not in seen:
            seen.add(source)
            unique_sources.append(source)

    if not unique_sources:
        return answer

    citation_lines = [f"[{idx}] {src}" for idx, src in enumerate(unique_sources, start=1)]
    return f"{answer}\n\nSources:\n" + "\n".join(citation_lines)

# --- Shared context retriever using Gemini graph ---
# --- Shared context retriever using Gemini graph ---
# --- Shared context retriever using Gemini graph ---
async def run_gemini_branch(question: str, thread_id: str) -> Dict[str, Any]:
    """Run Gemini-based RAG workflow and return full final state."""
    config = {"configurable": {"thread_id": thread_id}}

    result = None
    try:
        # Prefer full async invoke to get final state (includes metadata)
        # If your LangGraph version doesn't support ainvoke, fallback to invoke
        try:
            result = await app.ainvoke({"question": question}, config=config)
        except AttributeError:
            # synchronous fallback if async API not available
            result = app.invoke({"question": question}, config=config)
    except Exception as e:
        print(f"[run_gemini_branch] ⚠️ Error on ainvoke/invoke: {e}")
        result = None
        # fallback: stream and take the last emitted state
        try:
            for output in app.stream({"question": question}, config=config):
                for _, value in output.items():
                    result = value
        except Exception as ex:
            print(f"[run_gemini_branch] ⚠️ Stream fallback failed: {ex}")
            result = None

    # If result exists and rewrite is inside rewrite_metadata, normalize top-level key
    if isinstance(result, dict):
        # Some graph runtimes may include rewrite inside rewrite_metadata only
        md = result.get("rewrite_metadata") or {}
        if "rewritten_query" in md and not result.get("rewritten_query"):
            result["rewritten_query"] = md.get("rewritten_query")

    print(f"[run_gemini_branch] ✅ Final state keys: {list(result.keys()) if result else 'None'}")
    return result or {}



# --- Gemini Generation (RAG-based) ---
async def generate_with_gemini(question: str, documents, messages):
    gemini_llm = get_llm_model()
    system_prompt = """You are an expert assistant. 
                    Use the retrieved context to answer the question clearly and concisely. 
                    If you don't know, say so. Use at most three sentences.
                    Question: {question}
                    Context: {context}
                    Answer:"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | gemini_llm | StrOutputParser()

    context = "\n\n".join([doc.page_content for doc in documents])
    history = "\n".join([f"{m.type}: {m.content}" for m in messages[-10:]])
    answer = chain.invoke({"context": context, "question": question, "history": history})
    return answer

# --- OpenAI Generation (parallel answer) ---
async def generate_with_openai(question: str, documents, messages):
    openai_llm = get_openai_validator()  # GPT-4o-mini
    system_prompt = """You are an assistant tasked with providing factual, context-aware answers. 
Use the retrieved context to answer the question. Keep it short (max 3 sentences).
Question: {question}
Context: {context}
Answer:"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | openai_llm | StrOutputParser()

    context = "\n\n".join([doc.page_content for doc in documents])
    history = "\n".join([f"{m.type}: {m.content}" for m in messages[-10:]])
    answer = chain.invoke({"context": context, "question": question, "history": history})
    return answer

# --- Perplexity as Judge ---
async def judge_with_perplexity(question: str, gemini_answer: str, openai_answer: str) -> tuple[str, str]:
    perplexity_llm = get_perplexity_llm()

    system_prompt = """You are a factual evaluator.
        Compare these two answers to the given question and decide which one is more accurate, clear, and relevant.
        Return your judgment in this format:
        Selected: [Gemini or OpenAI]
        Reason: [short explanation]

        Question: {question}
        Gemini Answer: {gemini_answer}
        OpenAI Answer: {openai_answer}"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | perplexity_llm | StrOutputParser()

    validation = chain.invoke({
        "question": question,
        "gemini_answer": gemini_answer,
        "openai_answer": openai_answer,
    })

    # Parse output
    lines = validation.split("\n")
    selected = "Gemini" if "Gemini" in (lines[0] if lines else "") else "OpenAI"
    reason = lines[1].replace("Reason:", "").strip() if len(lines) > 1 else "Reason not clear."
    return selected, reason

# --- Parallel workflow orchestrator ---
async def run_parallel_branches(question: str, thread_id: str, state: GraphState):
    """Run Gemini and OpenAI branches concurrently with grading before judgment."""

    # --- Step 1: Context Retrieval ---
    gemini_result = await run_gemini_branch(question, thread_id)
    documents = gemini_result.get("documents", [])
    messages = gemini_result.get("messages", [])

    # --- Step 2: Generate in parallel (Gemini + OpenAI) ---
    gemini_task = asyncio.create_task(generate_with_gemini(question, documents, messages))
    openai_task = asyncio.create_task(generate_with_openai(question, documents, messages))
    gemini_answer, openai_answer = await asyncio.gather(gemini_task, openai_task)
    
    # ✅ Rewritten query from pipeline
    rewritten_query = gemini_result.get("rewritten_query", question)
    print(f"[parallel_runner] ✅ Rewritten Query (for UI): {repr(rewritten_query)}")

    # --- Step 3: Run Hallucination & Relevance Graders ---
    doc_text = "\n\n".join([doc.page_content for doc in documents])

    gemini_hallucination = hallucination_grader.invoke({
        "documents": doc_text,
        "generation": gemini_answer
    }).binary_score

    openai_hallucination = hallucination_grader.invoke({
        "documents": doc_text,
        "generation": openai_answer
    }).binary_score
    
    gemini_relevance = answer_grader.invoke({
        "question": question,
        "generation": gemini_answer
    }).binary_score

    openai_relevance = answer_grader.invoke({
        "question": question,
        "generation": openai_answer
    }).binary_score

    # --- Step 3.5: Evaluate retrieval metrics & confidence ---
    # Import grader directly from your workflow
    from src.workflow.graph import grade_generation_grounded_in_documents_and_question

    # Prepare the current graph state for evaluation
    state.update({
        "question": question,
        "documents": documents,
        "generation": gemini_answer
    })

    # Compute retrieval metrics + confidence (fills context_relevance, precision, confidence)
    grade_generation_grounded_in_documents_and_question(state)

    # --- Step 4: Apply grading filters ---
    if not gemini_hallucination or not gemini_relevance:
        gemini_answer = "[❌ Gemini answer flagged: hallucination or irrelevant]"
    if not openai_hallucination or not openai_relevance:
        openai_answer = "[❌ OpenAI answer flagged: hallucination or irrelevant]"

    # --- Step 5: Judge between valid ones ---
    selected, reason = await judge_with_perplexity(question, gemini_answer, openai_answer)
    final_answer = gemini_answer if selected == "Gemini" else openai_answer
    final_answer = build_cited_answer(final_answer, documents)

    # --- Step 6: Log everything ---
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "rewritten_query": rewritten_query,
        "gemini_answer": gemini_answer,
        "openai_answer": openai_answer,
        "gemini_hallucination": gemini_hallucination,
        "openai_hallucination": openai_hallucination,
        "gemini_relevance": gemini_relevance,
        "openai_relevance": openai_relevance,
        "selected": selected,
        "reason": reason,
        "sources": [doc.metadata.get("source", "Unknown Source") for doc in documents],
        # ✅ metrics now computed by the grader
        "context_relevance": state.get("context_relevance", 0.0),
        "context_precision": state.get("context_precision", 0.0),
        "confidence": state.get("confidence", 0.0),
        "retriever_type": gemini_result.get("retriever_type", "vectorstore"),
        "docs_used": len(documents)
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return final_answer, reason, selected, documents, rewritten_query
