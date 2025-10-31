"""
Hallucination Grader
--------------------
Evaluates whether a generated answer is factually grounded in retrieved documents.
Uses Gemini 4.0 (or 2.5 Flash) via structured output with a binary True/False response.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from src.models.model import get_llm_model


# ✅ Define structured output model
class GradeHallucinations(BaseModel):
    """Binary score indicating whether the answer is grounded in the provided documents."""
    binary_score: bool = Field(description="True if answer is grounded in facts, False otherwise.")


# ✅ Build the hallucination grader chain
def build_hallucination_grader():
    llm = get_llm_model()

    # Use structured output model
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt for grounding evaluation
    system_prompt = """You are a grader assessing whether an LLM's generation is grounded in or supported by 
    the provided set of retrieved documents. 
    Respond with a boolean:
    - True → The answer is grounded in the facts.
    - False → The answer includes hallucinations or unsupported statements."""

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Set of facts:\n\n{documents}\n\nGenerated answer:\n{generation}")
    ])

    # ✅ Combine into a runnable pipeline (modern syntax)
    chain = hallucination_prompt | structured_llm_grader

    # ✅ Wrap into a safe object with invoke()
    class HallucinationGrader:
        def invoke(self, inputs):
            try:
                result = chain.invoke(inputs)
                if result is None:
                    raise ValueError("Empty result from model")
                return result
            except Exception as e:
                print(f"[hallucination_grader] ⚠️ Error during grading: {e}")
                # Always return a safe object
                return GradeHallucinations(binary_score=False)

    return HallucinationGrader()


# ✅ Export ready-to-use grader
hallucination_grader = build_hallucination_grader()
