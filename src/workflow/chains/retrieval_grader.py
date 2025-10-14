from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.models.model import get_llm_model

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm = get_llm_model()

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains keywords, concepts, or semantic meaning related to the question grade it as relevant ('yes'), even if only partially relevant. Be very lenient to retain documents that might contribute to answering the question. Only grade as 'no' if the document is clearly unrelated to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Question: {question}\n\nDocument: {document}\n\nIs this document relevant to the question? Provide a binary score: 'yes' or 'no'.",
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader