from typing import Any, Dict, Set
from src.workflow.chains.retrieval_grader import retrieval_grader
from src.workflow.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates document relevance to the question using the retrieval grader.
    Skips invalid or short documents and ensures robust handling of grader responses.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state.get("question", "").strip()
    documents = state.get("documents", [])
    unique_sources: Set[str] = state.get("unique_sources", set())

    filtered_docs = []
    relevant_count = 0
    total_docs = len(documents)

    if not documents:
        print("---No documents provided, routing to web search---")
        return {
            "documents": [],
            "question": question,
            "web_search": True,
            "unique_sources": unique_sources,
            "retry_count": state.get("retry_count", 0),
        }

    for i, d in enumerate(documents):
        try:
            source = getattr(d, "metadata", {}).get("source", "Unknown Source")
            content = getattr(d, "page_content", "").strip()

            # 🧩 Skip empty or trivial documents
            if len(content) < 100:
                print(f"---Skipping short or empty doc {i} from {source}---")
                continue

            # 🧠 Grade document relevance
            score = retrieval_grader.invoke({"question": question, "document": content})

            # 🧱 Handle missing / invalid grader outputs gracefully
            if not score or not hasattr(score, "binary_score"):
                print(f"---DOC {i} GRADER RETURNED NONE OR INVALID FORMAT---")
                continue

            value = score.binary_score
            # Normalize to boolean safely
            if isinstance(value, str):
                value = value.strip().lower() in ["yes", "true", "1"]

            print(f"---DOC {i} SCORE: {value}---")

            if value:
                filtered_docs.append(d)
                unique_sources.add(source)
                relevant_count += 1

        except Exception as e:
            print(f"---DOC {i} GRADING FAILED: {e}---")
            continue

    # 🧭 Routing decision
    if relevant_count == 0:
        print("---No relevant docs found → switching to websearch---")
        web_search = True
    else:
        web_search = False
        print(f"---{relevant_count}/{total_docs} docs relevant → staying in vectorstore---")

    # 🧹 Keep top 5 max docs for generation
    documents = filtered_docs[:5]

    print(f"---FINAL DECISION (by model): web_search={web_search}, kept {len(documents)} docs---")

    return {
        "documents": documents,
        "question": question,
        "web_search": web_search,
        "unique_sources": unique_sources,
        "retry_count": state.get("retry_count", 0),
    }
