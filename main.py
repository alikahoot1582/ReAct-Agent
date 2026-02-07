import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia

load_dotenv()

# ---------------- UI Setup ----------------
st.set_page_config(page_title="ReAct Explorer", page_icon="⚖️", layout="wide")

# Minimalist Elegant CSS (Banner and Black Card Only)
st.markdown("""
    <style>
    .main-banner {
        background: #0f172a;
        color: #ffffff;
        padding: 25px;
        text-align: center;
        border-radius: 12px;
        font-size: 26px;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }
    .output-card {
        background-color: #111111;
        color: #00ff41; /* Matrix/Terminal Green for contrast, or #ffffff for white */
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #e94560;
        font-family: 'Consolas', monospace;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background: #0f172a; color: #94a3b8;
        text-align: center; padding: 10px; font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Tool Functions ----------------
def live_web_search(query):
    """Executes a live search via DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=3))
            if not results: return "No live web results found."
            return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
    except Exception as e:
        return f"Web search error: {e}"

def wiki_search(query):
    """Fetches summary from Wikipedia."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception:
        return "No Wikipedia page found for this query."

# ---------------- Agent Logic ----------------
def mini_agent(client, model, question):
    # System Prompt instructing the model on how to use the information
    SYSTEM_PROMPT = """You are a high-level research intelligence. 
    You have access to live web data and Wikipedia summaries.
    Structure your response with:
    1. Think: Brief strategy.
    2. Findings: Synthesized data from search.
    3. Final Answer: Your professional conclusion.
    """
    
    with st.status("🚀 Accessing Knowledge Streams...", expanded=True) as status:
        st.write("Searching Wikipedia...")
        wiki_data = wiki_search(question)
        
        st.write("Performing Live Web Search...")
        web_data = live_web_search(question)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Combine data for the LLM
    context = f"Wikipedia: {wiki_data}\n\nWeb Search: {web_data}"
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2
        )
        
        full_response = resp.choices[0].message.content

        # Display Final Synthesis in the Black Card
        st.subheader("🎯 Research Intelligence Output")
        st.markdown(f"""
            <div class="output-card">
                {full_response.replace(chr(10), '<br>')}
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Reasoning Error: {e}")

# ---------------- Main UI ----------------
st.markdown('<div class="main-banner">RE-ACT KNOWLEDGE EXPLORER</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password")
model_name = st.sidebar.selectbox("Engine", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

# User Input
query = st.chat_input("What would you like to research?")

if query:
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to begin.")
    else:
        client = Groq(api_key=api_key)
        mini_agent(client, model_name, query)

# Footer
st.markdown('<div class="footer">MADE BY MUHAMMAD ALI KAHOOT • 2026</div>', unsafe_allow_html=True)
