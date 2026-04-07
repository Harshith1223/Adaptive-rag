# Adaptive RAG System

An **adaptive, Retrieval-Augmented Generation (RAG) system** that dynamically rewrites queries, routes requests, retrieves context, evaluates grounding, and compares multiple LLM outputs using structured grading and confidence metrics.

This project is designed to be **transparent, self-evaluating, and extensible**, avoiding black-box LLM behavior.

---

## 🚀 Key Features

- 🧠 **Agentic Workflow (LangGraph)**
  - Query rewriting
  - Query analysis
  - Source routing (vector / web)
  - Iterative retry with confidence thresholds

- 🔀 **Multi-LLM Parallel Generation**
  - Gemini-based RAG generation
  - OpenAI-based parallel generation
  - Perplexity LLM used as an independent judge

- 📚 **Retrieval-Aware Evaluation**
  - Hybrid retrieval (dense similarity + MMR + lexical reranking)
  - Context relevance (semantic similarity)
  - Context precision (relevant document ratio)
  - Answer relevance
  - Hallucination detection
  - Weighted confidence score

- ⚖️ **Self-Grading & Selection**
  - Answers are filtered if hallucinated or irrelevant
  - Only valid answers are compared
  - Best answer is selected with a reason

- 🧩 **Web-to-Vector Memory**
  - Fresh web search results are chunked and added to Chroma
  - Future vector retrieval can reuse that learned context

- 🛠️ **MCP Tool Orchestration**
  - Tool-routed queries execute through a dedicated tool orchestrator node
  - Supports domain tools (e.g., weather / stock) with retrieval-ready outputs

- 🪵 **Full Observability**
  - Every query logs:
    - rewritten query
    - model answers
    - hallucination flags
    - relevance flags
    - context relevance
    - context precision
    - confidence score
    - selected model & reason

---

## 🏗️ System Architecture
```

User Query  
↓  
Query Rewriter  
↓  
Query Analyzer  
↓  
Source Router ───▶ Web Search (if needed)  
↓  
Vector Retrieval  
↓  
Document Grading  
↓  
Parallel Generation  
├─ Gemini (RAG)  
└─ OpenAI  
↓  
Hallucination + Relevance Graders  
↓  
Retrieval Metrics + Confidence Scoring  
↓  
Perplexity Judge  
↓  
Final Answer + Logs

```yaml
---

## 🧠 What Makes This “Agentic”?

This system:
- Maintains **state across steps**
- Makes **decisions based on evaluation results**
- Retries generation if confidence is low
- Chooses tools and sources dynamically
- Grades its own outputs before returning answers

This is **not just RAG**, but an **autonomous reasoning pipeline**.

---

## 📊 Evaluation Metrics

| Metric | Description |
|------|------------|
| `answer_relevance` | Does the answer address the question? |
| `factual_consistency` | Is the answer grounded in retrieved documents? |
| `context_relevance` | Semantic similarity between query and retrieved docs |
| `context_precision` | Fraction of highly relevant documents |
| `confidence` | Weighted score combining all metrics |

### Confidence Formula
```

confidence =  
0.4 × answer\_relevance

-   0.3 × factual\_consistency
    
-   0.2 × context\_relevance
    
-   0.1 × context\_precision
    

```yaml
---

## 🧪 Logged Output Example

```json
{
  "question": "What is an AI Agent?",
  "rewritten_query": "Definition and characteristics of an AI agent",
  "gemini_answer": "...",
  "openai_answer": "...",
  "gemini_hallucination": true,
  "openai_hallucination": true,
  "context_relevance": 0.82,
  "context_precision": 0.67,
  "confidence": 0.78,
  "selected": "OpenAI",
  "reason": "Clearer explanation with concrete examples",
  "docs_used": 4
}
```

---

## 🧰 Tech Stack

-   **Python 3.10+**
    
-   **LangGraph** – agentic workflow orchestration
    
-   **LangChain** – prompts, chains, evaluators
    
-   **SentenceTransformers / HuggingFace** – embeddings
    
-   **Gemini** – primary RAG generator
    
-   **OpenAI (GPT-4o-mini)** – parallel generation
    
-   **Perplexity LLM** – judge / evaluator
    
-   **NLTK / BERTScore** – evaluation utilities
    
-   **Streamlit / API layer** – UI (optional)
    

---

## 📁 Project Structure

```bash
src/
├── workflow/
│   ├── graph.py              # LangGraph workflow
│   ├── state.py              # GraphState definition
│   ├── nodes/                # retrieve, generate, web_search, etc.
│   ├── chains/
│   │   ├── hallucination_grader.py
│   │   ├── answer_grader.py
│   │   └── router.py
├── models/
│   ├── model.py              # Gemini, OpenAI, Perplexity loaders
logs/
│   └── model_selection_log.txt
```

---

## 🎯 Why This Project Matters

Most RAG systems:

-   Return answers without verification
    
-   Hide failure modes
    
-   Cannot explain *why* an answer was chosen
    

This system:

-   **Evaluates itself**
    
-   **Logs every decision**
    
-   **Quantifies confidence**
    
-   **Chooses answers rationally**
    

It is suitable for:

-   Research
    
-   Production experimentation
    
-   Agentic AI learning
    
-   Resume / portfolio demonstration
    

---

## 🔮 Future Enhancements

-   Per-model confidence comparison (Gemini vs OpenAI)
    
-   Tool-using agents
    
-   Multi-hop retrieval
    
-   Feedback-based learning
    
-   Evaluation dashboards
    
-   MCP / A2A agent communication
    

---

## 📜 License

MIT License

---
