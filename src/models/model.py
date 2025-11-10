from dotenv import load_dotenv
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_perplexity import ChatPerplexity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --------------------------------------------------------------------- #
# 🧠 Gemini 2.5 Flash Loader (with Retry + Backoff)
# --------------------------------------------------------------------- #
def get_llm_model(max_retries: int = 5, base_delay: float = 2.0):
    """
    Returns a stable Gemini 2.5 Flash model with built-in retry + exponential backoff
    to handle transient 429 (ResourceExhausted) errors gracefully.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY missing from .env file")

    attempt = 0
    while attempt < max_retries:
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # ⚡ Fast and reasoning-capable
                temperature=0.3,
                max_output_tokens=1024,
                convert_system_message_to_human=True,
                google_api_key=api_key,
            )
            # quick smoke test to ensure model instantiates correctly
            _ = model.model
            if attempt > 0:
                print(f"[Gemini] ✅ Recovered after {attempt} retries.")
            return model

        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait_time = base_delay * (2 ** attempt)
                print(f"[Gemini] ⚠️ Rate limit (429) hit. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                print(f"[Gemini] ❌ Initialization failed: {e}")
                raise

    # if still failing after retries
    print("[Gemini] ❌ Max retries exceeded. Falling back to minimal model configuration.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # lighter fallback
        temperature=0.4,
        max_output_tokens=512,
        google_api_key=api_key,
    )


# --------------------------------------------------------------------- #
# 🧠 Perplexity, OpenAI, Embedding, and Validator Loaders
# --------------------------------------------------------------------- #
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

    system_prompt = """Compare these two answers to the question.
Return output in this format:
Selected: [Gemini or Perplexity]
Reason: [Brief explanation]"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | validator_llm | StrOutputParser()
    return chain
