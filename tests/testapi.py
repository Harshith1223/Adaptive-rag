from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_perplexity import ChatPerplexity
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("Loaded keys:")
print("OPENAI_API_KEY:", (OPENAI_API_KEY[:10] + "...") if OPENAI_API_KEY else "❌ Missing")
print("PERPLEXITY_API_KEY:", (PERPLEXITY_API_KEY[:10] + "...") if PERPLEXITY_API_KEY else "❌ Missing")
print("TAVILY_API_KEY:", (TAVILY_API_KEY[:10] + "...") if TAVILY_API_KEY else "❌ Missing")

# --- Test OpenAI ---
print("\n🔍 Testing OpenAI (GPT-4o-mini)...")
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    response = llm.invoke("Hello, test OpenAI connection.")
    print("✅ OpenAI Success:", response.content)
except Exception as e:
    print("❌ OpenAI Error:", str(e))

# --- Test Perplexity ---
print("\n🔍 Testing Perplexity (Sonar)...")
try:
    llm = ChatPerplexity(model="sonar", temperature=0, api_key=PERPLEXITY_API_KEY)
    response = llm.invoke("Hello, test Perplexity connection.")
    print("✅ Perplexity Success:", response.content)
except Exception as e:
    print("❌ Perplexity Error:", str(e))

# --- Test Tavily ---
print("\n🔍 Testing Tavily Search...")
try:
    search = TavilySearch(max_results=1, api_key=TAVILY_API_KEY)
    result = search.invoke({"query": "test"})
    print("✅ Tavily Success:", result)
except Exception as e:
    print("❌ Tavily Error:", str(e))
