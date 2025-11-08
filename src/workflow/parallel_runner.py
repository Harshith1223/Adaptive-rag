import asyncio
from typing import Dict, Any
from src.workflow.graph import app
from src.models.model import get_perplexity_llm, get_llm_model, get_openai_validator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, os
from datetime import datetime
from src.workflow.chains.hallucination_grader import hallucination_grader  # ✅ import your grader
from src.workflow.chains.answer_grader import answer_grader                # ✅ import second grader

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "model_selection_log.txt")


# --- Shared context retriever using Gemini graph ---
async def run_gemini_branch(question: str, thread_id: str) -> Dict[str, Any]:
    """Run Gemini-based RAG workflow."""
    config = {"configurable": {"thread_id": thread_id}}
    result = None
    for output in app.stream({"question": question}, config=config):
        for _, value in output.items():
            result = value
    
    if result and "generation" in result:
        print(f"\n🧩 Adaptive RAG Generated Answer:\n{result['generation']}\n")
        
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
async def run_parallel_branches(question: str, thread_id: str):
    """Run Gemini and OpenAI branches concurrently with grading before judgment."""

    # --- Step 1: Context Retrieval ---
    gemini_result = await run_gemini_branch(question, thread_id)
    documents = gemini_result.get("documents", [])
    messages = gemini_result.get("messages", [])

    # --- Step 2: Generate in parallel (Gemini + OpenAI) ---
    gemini_task = asyncio.create_task(generate_with_gemini(question, documents, messages))
    openai_task = asyncio.create_task(generate_with_openai(question, documents, messages))
    gemini_answer, openai_answer = await asyncio.gather(gemini_task, openai_task)

    # --- Step 3: Run Hallucination & Relevance Graders ---
    doc_text = "\n\n".join([doc.page_content for doc in documents])

    # 🧠 Hallucination checks
    gemini_hallucination = hallucination_grader.invoke({
        "documents": doc_text,
        "generation": gemini_answer
    }).binary_score

    openai_hallucination = hallucination_grader.invoke({
        "documents": doc_text,
        "generation": openai_answer
    }).binary_score

    # ✅ Answer relevance checks
    gemini_relevance = answer_grader.invoke({
        "question": question,
        "generation": gemini_answer
    }).binary_score

    openai_relevance = answer_grader.invoke({
        "question": question,
        "generation": openai_answer
    }).binary_score

    # --- Step 4: Apply grading filter ---
    if not gemini_hallucination or not gemini_relevance:
        gemini_answer = "[❌ Gemini answer flagged: hallucination or irrelevant]"
    if not openai_hallucination or not openai_relevance:
        openai_answer = "[❌ OpenAI answer flagged: hallucination or irrelevant]"

    # --- Step 5: Judge between valid ones ---
    selected, reason = await judge_with_perplexity(question, gemini_answer, openai_answer)
    final_answer = gemini_answer if selected == "Gemini" else openai_answer

    # --- Step 6: Log everything ---
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "gemini_answer": gemini_answer,
        "openai_answer": openai_answer,
        "gemini_hallucination": gemini_hallucination,
        "openai_hallucination": openai_hallucination,
        "gemini_relevance": gemini_relevance,
        "openai_relevance": openai_relevance,
        "selected": selected,
        "reason": reason,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return final_answer, reason, selected, documents
    