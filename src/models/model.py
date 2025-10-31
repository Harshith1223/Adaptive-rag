from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  
from langchain_perplexity import ChatPerplexity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
def get_llm_model():
    """
    Returns a stable Gemini 2.5 Flash model for text-based tasks.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY missing from .env file")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # ⚡ Fast and reasoning-capable
        temperature=0.3,
        max_output_tokens=1024,
        convert_system_message_to_human=True,
        google_api_key=api_key,  # ✅ force correct endpoint
    )


def get_perplexity_llm():
    return ChatPerplexity(
        model="sonar",
        temperature=0,
        api_key=os.getenv("PERPLEXITY_API_KEY"), 
    )

def get_openai_validator():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_embed_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

def get_validator_chain():
    validator_llm = get_openai_validator()

    system_prompt = """Compare these two answers to the question...
    Output format: Selected: [Gemini or Perplexity]
    Reason: [Brief explanation]"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | validator_llm | StrOutputParser()
    return chain
