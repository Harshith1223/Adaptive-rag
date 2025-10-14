from typing import Any, Dict, Set
from src.workflow.chains.retrieval_grader import retrieval_grader
from src.workflow.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates document relevance to the question.
    Leaves routing decisions entirely to the router agent.
    Only sets web_search=True if absolutely no relevant documents exist.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    unique_sources: Set[str] = state.get("unique_sources", set())

    filtered_docs = []
    relevant_count = 0
    total_docs = len(documents)

    for i, d in enumerate(documents):
        source = d.metadata.get("source", "Unknown Source")
        content = d.page_content.strip()

        if len(content) < 100:
            print(f"---Skipping short doc {i} from {source}---")
            continue

        score = retrieval_grader.invoke({"question": question, "document": content})
        print(f"---DOC {i} SCORE: {score.binary_score}---")

        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
            unique_sources.add(source)
            relevant_count += 1

    # Only set web_search=True if no relevant docs at all
    if relevant_count == 0:
        print("---No relevant docs found → switching to websearch---")
        web_search = True
    else:
        web_search = False
        print(f"---{relevant_count}/{total_docs} docs relevant → staying in vectorstore---")

    documents = filtered_docs[:5]

    print(f"---FINAL DECISION (by model): web_search={web_search}, kept {len(documents)} docs---")

    return {
        "documents": documents,
        "question": question,
        "web_search": web_search,
        "unique_sources": unique_sources,
        "retry_count": state.get("retry_count", 0),
    }
