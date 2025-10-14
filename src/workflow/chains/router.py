from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.models.model import get_llm_model

class RouteQuery(BaseModel): #output format
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Choose 'vectorstore' for AI/LLM topics, otherwise 'websearch'.",
    )

llm = get_llm_model()

structured_llm_router = llm.with_structured_output(RouteQuery)

# ------------- STRONGER REASONING LOGIC -------------
system = """You are an expert router deciding whether a question should be answered
using an internal vectorstore or via live web search.

The vectorstore ONLY contains documents about:
- AI agents and autonomous LLM systems
- prompt engineering and instruction tuning
- adversarial attacks and LLM safety topics

If the user's question is about general knowledge, or anything not listed above**,
route it to **websearch**.

Be strict:
- Only AI-related questions → vectorstore
- Everything else → websearch
"""

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

question_router = route_prompt | structured_llm_router
