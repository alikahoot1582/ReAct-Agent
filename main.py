import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia
import arxiv
from deep_translator import GoogleTranslator
import sympy
from transformers import pipeline

load_dotenv()

# ---------------- UI Setup ----------------
st.set_page_config(page_title="ReAct Knowledge Explorer", page_icon="⚖️", layout="wide")

# Custom Elegant CSS
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
        padding: 40px;
        margin: 20px auto;
        max-width: 1000px;
        font-size: 28px;
        font-weight: 300;
        text-align: center;
        border-radius: 20px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        letter-spacing: 2px;
    }
    .step-pill {
        background: #ffffff;
        color: #64748b;
        padding: 10px 30px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
        width: fit-content;
        margin: -25px auto 30px auto;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        letter-spacing: 1px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #0f172a;
        color: #94a3b8;
        text-align: center;
        padding: 15px;
        font-size: 12px;
        letter-spacing: 3px;
        z-index: 1000;
        border-top: 1px solid #334155;
    }
    .stExpander { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Intelligence Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY", "")
model_name = st.sidebar.selectbox(
    "Engine Selection",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0
)

# ---------------- Banner ----------------
st.markdown('<div class="main-banner">RE-ACT KNOWLEDGE EXPLORER</div>', unsafe_allow_html=True)
st.markdown('<div class="step-pill">THINK • ACT • OBSERVE • CONCLUDE</div>', unsafe_allow_html=True)

query = st.chat_input("Enter your research objective...")

# ---------------- Prompt ----------------
# The critical fix: "DO NOT use built-in tools"
SYSTEM_PROMPT = """
You are a world-class research assistant. 
IMPORTANT: You must respond ONLY in plain text. DO NOT attempt to use any built-in functions or tool-calling features.

For every question, you MUST output exactly these five sections in order:

Step 1 - Think: <Internal thought process>
Step 2 - Observe: <Current state of knowledge>
Step 3 - Act: <Choose one: WebSearch, Wikipedia, Math, News, Books, Translate, Sentiment, Arxiv>
Step 4 - Conclude: <Summary of findings>

Final Answer: <Clear, professional conclusion>

Follow this format strictly.
"""

# ---------------- Agent Logic ----------------
def mini_agent(client, model, question):
    try:
        # We explicitly set tool_choice to None (if supported) or just rely on the prompt
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Research Question: {question}"},
            ],
            temperature=0.1, # Lower temperature for better formatting adherence
            max_tokens=1000
        )

        content = resp.choices[0].message.content or ""

        # Extraction logic with fallback to "Not specified"
        def extract(pattern, text):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else "Information not captured in this step."

        thought = extract(r"Step 1 - Think:(.*?)Step 2", content)
        observe = extract(r"Step 2 - Observe:(.*?)Step 3", content)
        act = extract(r"Step 3 - Act:(.*?)Step 4", content)
        conclude = extract(r"Step 4 - Conclude:(.*?)Final Answer", content)
        final = extract(r"Final Answer:(.*)", content)

        # UI Display
        with st.expander("🔍 VIEW REASONING CHAIN", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Step 1: Thought**\n\n{thought}")
                st.markdown(f"**Step 2: Observation**\n\n{observe}")
            with cols[1]:
                st.markdown(f"**Step 3: Action**\n\n{act}")
                st.markdown(f"**Step 4: Conclusion**\n\n{conclude}")

        st.markdown("---")
        st.markdown("### 🎯 Final Synthesis")
        st.write(final)

    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        if "tool_use_failed" in str(e):
            st.warning("The model attempted to trigger an internal tool call. I've blocked the request to prevent a crash. Please try rephrasing or using a different model.")

# ---------------- Run ----------------
if query:
    st.chat_message("user").write(query)
    if not api_key:
        st.error("Missing Credentials: Enter your Groq API Key in the sidebar.")
    else:
        client = Groq(api_key=api_key)
        with st.spinner("Executing Research Protocol..."):
            mini_agent(client, model_name, query)

# ---------------- Footer ----------------
st.markdown(
    """
    <div class="footer">
        MADE BY MUHAMMAD ALI KAHOOT • 2026
    </div>
    """,
    unsafe_allow_html=True
)
