import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------- UI Setup ----------------
st.set_page_config(page_title="ReAct Knowledge Explorer", page_icon="⚖️", layout="wide")

# Custom CSS for Black Output Styling
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f8fafc; }

    /* Elegant Banner */
    .main-banner {
        background: #0f172a;
        color: #ffffff;
        padding: 30px;
        text-align: center;
        border-radius: 15px;
        font-size: 24px;
        letter-spacing: 2px;
        font-weight: 300;
        margin-bottom: 40px;
    }

    /* THE BLACK OUTPUT BOX */
    .output-card {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #e94560;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
    }

    .step-label {
        color: #94a3b8;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.8rem;
        margin-bottom: 5px;
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #0f172a;
        color: #64748b;
        text-align: center;
        padding: 10px;
        font-size: 11px;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Agent Logic ----------------
def mini_agent(client, model, question):
    # (Simplified for demonstration of the Black UI output)
    SYSTEM_PROMPT = "You are a researcher. Provide steps: Think, Observe, Act, Conclude, and Final Answer."
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            temperature=0.1
        )
        content = resp.choices[0].message.content

        # Parsing (Simplified Regex)
        final_answer = content.split("Final Answer:")[-1] if "Final Answer:" in content else content

        # DISPLAY BLACK OUTPUT
        st.markdown("### 🎯 Final Synthesis")
        st.markdown(f"""
            <div class="output-card">
                <div class="step-label">Research Conclusion</div>
                {final_answer}
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- Main UI ----------------
st.markdown('<div class="main-banner">RE-ACT KNOWLEDGE EXPLORER</div>', unsafe_allow_html=True)

api_key = st.sidebar.text_input("Groq API Key", type="password")
model_name = st.sidebar.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
query = st.chat_input("Enter your query...")

if query:
    if not api_key:
        st.error("Please provide API Key")
    else:
        client = Groq(api_key=api_key)
        mini_agent(client, model_name, query)

st.markdown('<div class="footer">MADE BY MUHAMMAD ALI KAHOOT • 2026</div>', unsafe_allow_html=True)
