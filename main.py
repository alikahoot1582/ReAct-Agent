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

# Custom CSS for an Elegant Vibe
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Elegant Header Banner */
    .main-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e94560;
        padding: 30px;
        margin: 20px auto;
        max-width: 1000px;
        font-size: 24px;
        font-weight: 300;
        text-align: center;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(233, 69, 96, 0.2);
        letter-spacing: 1px;
    }

    /* Sophisticated Pill */
    .step-pill {
        background: #ffffff;
        color: #1a1a2e;
        padding: 8px 25px;
        border-radius: 50px;
        font-weight: 500;
        text-align: center;
        width: fit-content;
        margin: -15px auto 20px auto;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-family: 'Inter', sans-serif;
    }

    /* Footer Style */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #1a1a2e;
        color: #95a5a6;
        text-align: center;
        padding: 10px;
        font-weight: 400;
        font-size: 14px;
        letter-spacing: 2px;
        border-top: 1px solid #e94560;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY", "")
model_name = st.sidebar.selectbox(
    "Model Selection",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    index=1
)
max_steps = st.sidebar.slider("Reasoning Depth", 4, 10, 4)

# ---------------- Banner ----------------
st.markdown(
    '<div class="main-banner">RE-ACT KNOWLEDGE EXPLORER<br>'
    '<span style="font-size: 14px; color: #95a5a6; font-weight: 400;">'
    'Multi-Agent Research Intelligence Engine Engine</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="step-pill">THINKING • ACTING • OBSERVING • CONCLUDING</div>',
    unsafe_allow_html=True
)

st.markdown("### 🔍 Enterprise Search Query")
query = st.chat_input("Enter your research topic...")

# ---------------- Tool Functions ----------------
# (Functions remain the same as your original logic)
def tool_web_search(query, k=4):
    with DDGS() as ddg:
        results = ddg.text(query, region="us-en", max_results=k)
        return "\n".join([f"{r['title']} {r['href']}" for r in results if r]) or "No result found"

def tool_wikipedia(query, sentences=2):
    try:
        pages = wikipedia.search(query, results=1)
        if not pages: return "No Wikipedia page found"
        return wikipedia.summary(pages[0], sentences=sentences)
    except Exception as e:
        return str(e)

def tool_math(expr):
    return str(sympy.simplify(expr))

def tool_news(query):
    with DDGS() as ddg:
        results = ddg.news(query, region="us-en", max_results=3)
        return "\n".join([f"{r['title']} {r['url']}" for r in results if r]) or "No news found"

def tool_books(query):
    with DDGS() as ddg:
        results = ddg.text(query + " book", region="us-en", max_results=3)
        return "\n".join([f"{r['title']} {r['href']}" for r in results if r]) or "No books found"

def tool_translate(text):
    return GoogleTranslator(source="auto", target="ur").translate(text)

def tool_sentiment(text):
    return pipeline("sentiment-analysis")(text)[0]

def tool_arxiv(query):
    try:
        search = arxiv.Search(query=query, max_results=1)
        paper = next(search.results(), None)
        if not paper: return "No Arxiv paper found"
        return f"{paper.title}\n{paper.entry_id}"
    except Exception as e:
        return str(e)

# ---------------- Prompt ----------------
SYSTEM_PROMPT = """
You are a world-class research assistant.
For every question, you MUST output exactly four steps before the Final Answer:

- Step 1 - Think: Explain what you will do next.
- Step 2 - Observe: Record what information you have or tool output.
- Step 3 - Act: Choose ONE tool (WebSearch, Wikipedia, Math, News, Books, Translate, Sentiment, Arxiv) and provide the exact query or text.
- Step 4 - Conclude: Summarize what you learned.

Only after completing all four steps, give:
Final Answer: <short, clear, professional answer in English>
"""

# ---------------- Agent ----------------
def mini_agent(client, model, question):
    convo = SYSTEM_PROMPT + "\nUser Question: " + question

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": convo},
        ],
        temperature=0.2,
        max_tokens=800
    )

    text = resp.choices[0].message.content or ""

    # Regex for steps
    thought = re.search(r"Step 1 - Think:(.*)", text)
    observe = re.search(r"Step 2 - Observe:(.*)", text)
    act = re.search(r"Step 3 - Act:(.*)", text)
    conclude = re.search(r"Step 4 - Conclude:(.*)", text)
    final = re.search(r"Final Answer:(.*)", text)

    # UI Rendering
    with st.expander("📝 View Reasoning Process", expanded=True):
        st.write(f"**Step 1 - Think:** {thought.group(1).strip() if thought else '...'}")
        st.write(f"**Step 2 - Observe:** {observe.group(1).strip() if observe else '...'}")
        st.write(f"**Step 3 - Act:** {act.group(1).strip() if act else '...'}")
        st.write(f"**Step 4 - Conclude:** {conclude.group(1).strip() if conclude else '...'}")

    st.markdown(f"### 🎯 Final Synthesis")
    st.info(final.group(1).strip() if final else "No synthesis produced.")

# ---------------- Run ----------------
if query:
    st.chat_message("user").write(query)
    if not api_key:
        st.error("Authentication required: Please provide a valid GROQ API key.")
    else:
        client = Groq(api_key=api_key)
        with st.spinner("Analyzing data streams..."):
            mini_agent(client, model_name, query)

# ---------------- Elegant Footer ----------------
st.markdown(
    """
    <div class="footer">
        MADE BY MUHAMMAD ALI KAHOOT • 2024
    </div>
    """,
    unsafe_allow_html=True
)
