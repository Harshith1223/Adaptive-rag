import streamlit as st
from dotenv import load_dotenv
import asyncio
import os
import json
from datetime import datetime
from src.workflow.parallel_runner import run_parallel_branches
from src.workflow.graph import is_greeting
from src.cli.main import handle_greeting

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Adaptive RAG System", page_icon="🤖", layout="wide")

# --- Utility functions ---
def log_feedback(question, answer, rating):
    """Log user feedback to a file."""
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating
    }
    feedback_dir = "logs"
    feedback_file = os.path.join(feedback_dir, "feedback_log.txt")
    try:
        os.makedirs(feedback_dir, exist_ok=True)
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        return f"✅ Feedback logged successfully: {rating}"
    except Exception as e:
        return f"⚠️ Failed to log feedback: {str(e)}"

# --- Initialize session state ---
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "feedback_status" not in st.session_state:
    st.session_state.feedback_status = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_session_{datetime.now().timestamp()}"

# Title
st.title("🤖 Adaptive RAG System")

# Sidebar for query history
with st.sidebar:
    st.subheader("Query History")
    if st.session_state.query_history:
        for i, entry in enumerate(st.session_state.query_history):
            with st.expander(f"Q{i+1}: {entry['question']}", expanded=False):
                st.markdown(f"**Answer:** {entry['answer']}")
    else:
        st.info("No queries yet.")

# Input field
question = st.text_input("Enter your question:", placeholder="e.g., What are AI agents?")

col1, col2 = st.columns(2)
with col1:
    if st.button("Get Answer"):
        if not question.strip():
            st.error("Please enter a valid question.")
        else:
            try:
                with st.spinner("Processing (Gemini + OpenAI → Perplexity Judge)..."):
                    st.session_state.last_result = None
                    st.session_state.feedback_status = ""
                    st.session_state.current_answer = ""
                    st.session_state.current_question = question

                    thread_id = st.session_state.thread_id

                    # Greeting check
                    if is_greeting(question):
                        answer, _, sources = handle_greeting(question)
                        st.session_state.current_answer = answer
                        st.session_state.query_history.append({"question": question, "answer": answer})
                        st.markdown(f"**Answer:** {answer}")
                        st.success("Answer generated!")
                        st.subheader("Retrieved Documents")
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Doc {i}: {source}**")
                    else:
                        # Run full parallel workflow (Gemini + OpenAI judged by Perplexity)
                        final_answer, reason, selected, documents = asyncio.run(
                            run_parallel_branches(question, thread_id)
                            
                        )

                        # Try reading both answers from log file
                        gemini_answer, openai_answer = "", ""
                        try:
                            with open("logs/model_selection_log.txt", "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                if lines:
                                    import json
                                    record = json.loads(lines[-1])
                                    gemini_answer = record.get("gemini_answer", "")
                                    openai_answer = record.get("openai_answer", "")
                        except Exception:
                            pass

                        # --- Display all answers and decision ---
                        st.subheader("Gemini Answer")
                        st.write(gemini_answer or "_No Gemini answer available._")

                        st.subheader("OpenAI Answer")
                        st.write(openai_answer or "_No OpenAI answer available._")

                        st.markdown(f"### 🧠 Final Answer (from {selected})")
                        st.markdown(final_answer)
                        st.info(f"**Perplexity Decision:** {reason}")
                        st.success("Answer generated!")

                        # --- Display document links ---
                        if documents:
                            st.subheader("Retrieved Documents")
                            unique_sources = sorted(
                                set([doc.metadata.get("source", "Unknown Source") for doc in documents])
                            )
                            for i, source in enumerate(unique_sources, 1):
                                st.markdown(f"**Doc {i}:** [{source}]({source})")

                        # Save to session
                        st.session_state.current_answer = final_answer
                        st.session_state.query_history.append({"question": question, "answer": final_answer})
                        st.session_state.last_result = documents

            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    if st.button("Clear History"):
        st.session_state.query_history = []
        st.session_state.last_result = None
        st.session_state.feedback_status = ""
        st.session_state.current_answer = ""
        st.session_state.current_question = ""
        st.session_state.thread_id = f"user_session_{datetime.now().timestamp()}"
        st.rerun()

# Feedback buttons
if st.session_state.current_answer and st.session_state.current_question:
    st.markdown("**Rate this answer:**")
    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        if st.button("👍 Thumbs Up"):
            st.session_state.feedback_status = log_feedback(
                st.session_state.current_question,
                st.session_state.current_answer,
                "positive"
            )
    with fb_col2:
        if st.button("👎 Thumbs Down"):
            st.session_state.feedback_status = log_feedback(
                st.session_state.current_question,
                st.session_state.current_answer,
                "negative"
            )

# Show feedback status
if st.session_state.feedback_status:
    if "Failed" in st.session_state.feedback_status:
        st.warning(st.session_state.feedback_status)
    else:
        st.success(st.session_state.feedback_status)
