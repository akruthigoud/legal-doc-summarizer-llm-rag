bash

cat > /mnt/user-data/outputs/project5_ai_chatbot.py << 'PYEOF'
# ================================================================
# PROJECT 5 — AI Chatbot with Memory & Document QA
# Tech: LangChain + OpenAI + FAISS + Streamlit
# GitHub Repo Name: langchain-ai-chatbot
# ================================================================
# SETUP:
#   pip install streamlit langchain-openai langchain-community
#              langchain-text-splitters langchain-core faiss-cpu
#              openai pypdf python-dotenv
#   streamlit run project5_ai_chatbot.py
# ================================================================

import os, io
import streamlit as st
import pypdf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="ChatAI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0a0e1a;}
.header{background:linear-gradient(135deg,#1a1f3a,#0f1628);padding:24px 32px;
        border-radius:14px;border:1px solid #2a3050;margin-bottom:16px;}
.h-title{color:#38bdf8;font-size:26px;font-weight:700;margin:0;}
.h-sub{color:#94a3b8;font-size:13px;margin:4px 0 0;}
.user-msg{background:linear-gradient(135deg,#1e3a5f,#1a2f4a);border:1px solid #3b6ea5;
          border-radius:14px 14px 4px 14px;padding:13px 17px;margin:7px 0 7px 50px;
          color:#e2e8f0;line-height:1.75;font-size:14px;}
.bot-msg{background:#1a1f3a;border:1px solid #2a3050;border-radius:14px 14px 14px 4px;
         padding:13px 17px;margin:7px 50px 7px 0;color:#e2e8f0;line-height:1.75;font-size:14px;}
.lbl-user{color:#38bdf8;font-size:11px;font-weight:600;text-align:right;margin:3px 3px 0;}
.lbl-bot{color:#94a3b8;font-size:11px;font-weight:600;margin:0 0 3px 3px;}
.card{background:#1a1f3a;border:1px solid #2a3050;border-radius:10px;padding:14px 16px;margin:5px 0;}
.card-lbl{color:#38bdf8;font-size:11px;font-weight:600;}
.card-val{color:#e2e8f0;font-size:13px;margin-top:2px;}
.empty-state{text-align:center;padding:70px 0;color:#475569;}
.stButton>button{background:linear-gradient(135deg,#0ea5e9,#0284c7)!important;
                 color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;}
.stTextInput>div>input{background:#1a1f3a!important;color:#e2e8f0!important;
                       border-color:#2a3050!important;border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "mode" not in st.session_state:
    st.session_state.mode = "general"

# ── Core Functions ───────────────────────────────────────────
def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)

def load_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def build_vectorstore(text: str, api_key: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)

def build_history_string(messages: list) -> str:
    history = []
    for m in messages[-10:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history.append(f"{role}: {m['content']}")
    return "\n".join(history)

def chat_general(user_input: str, history: list, api_key: str) -> str:
    llm = get_llm(api_key)
    history_str = build_history_string(history)
    prompt = f"""You are ChatAI, a helpful and intelligent AI assistant.
You have memory of the conversation. Give clear, accurate, well-structured responses.

Conversation History:
{history_str}

User: {user_input}
ChatAI:"""
    return llm.invoke(prompt).content

def chat_with_doc(user_input: str, vectorstore: FAISS, api_key: str) -> str:
    llm = get_llm(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(user_input)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""You are a document expert assistant.
Answer using ONLY the provided document context.
If the answer is not in the document, say "This information is not in the uploaded document."

Document Context:
{context}

Question: {user_input}

Answer:"""
    return llm.invoke(prompt).content

QUICK_QUESTIONS_GENERAL = [
    "What is RAG (Retrieval-Augmented Generation)?",
    "Explain LangChain in simple terms",
    "How does prompt engineering work?",
    "What is the difference between CNN and LSTM?",
    "How do vector databases work?",
    "What is LangGraph used for?",
]

QUICK_QUESTIONS_DOC = [
    "Summarize this document",
    "What are the key points?",
    "Who are the main parties?",
    "What are the important dates?",
    "What obligations are listed?",
]

# ── UI ───────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <p class="h-title">🤖 ChatAI — Intelligent Assistant</p>
  <p class="h-sub">LangChain · OpenAI GPT · FAISS · Conversation Memory · Document Q&A</p>
</div>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY",""), placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.divider()

    st.markdown("### 💬 Mode")
    mode_choice = st.radio("Select:", ["General Chat", "Document QA"])
    st.session_state.mode = "general" if mode_choice == "General Chat" else "document"
    st.divider()

    if st.session_state.mode == "document":
        st.markdown("### 📄 Upload PDF")
        uploaded_doc = st.file_uploader("Upload PDF:", type=["pdf"])
        if uploaded_doc:
            if uploaded_doc.name != st.session_state.doc_name:
                if not api_key:
                    st.error("Enter API key first.")
                else:
                    with st.spinner("Indexing document..."):
                        text = load_pdf(uploaded_doc.read())
                        st.session_state.vectorstore = build_vectorstore(text, api_key)
                        st.session_state.doc_name = uploaded_doc.name
                    st.success(f"✅ '{uploaded_doc.name}' ready!")
        st.divider()

    st.markdown("### 💡 Quick Questions")
    questions = QUICK_QUESTIONS_DOC if st.session_state.mode == "document" else QUICK_QUESTIONS_GENERAL
    for q in questions:
        if st.button(q, key=f"qq_{hash(q)}", use_container_width=True):
            st.session_state["queued_q"] = q
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 📊 Session")
    st.markdown(f'<div class="card"><div class="card-lbl">Messages</div>'
                f'<div class="card-val">{len(st.session_state.messages)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="card-lbl">Mode</div>'
                f'<div class="card-val">{"Document QA" if st.session_state.mode=="document" else "General Chat"}</div></div>', unsafe_allow_html=True)
    if st.session_state.doc_name:
        st.markdown(f'<div class="card"><div class="card-lbl">Document</div>'
                    f'<div class="card-val">{st.session_state.doc_name}</div></div>', unsafe_allow_html=True)

# Chat window
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:56px">🤖</div>
        <h3 style="color:#94a3b8;margin-top:12px">Welcome to ChatAI!</h3>
        <p style="color:#64748b">Ask me anything about AI, ML, or upload a document for Q&A.<br>
        I remember our full conversation.</p>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown('<div class="lbl-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="lbl-bot">🤖 ChatAI</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Handle queued quick question
queued = st.session_state.pop("queued_q", "")

col_in, col_send = st.columns([6, 1])
with col_in:
    user_input = st.text_input(
        "Message",
        value=queued,
        placeholder="Type your message and press Enter or click Send...",
        label_visibility="collapsed",
        key="chat_input"
    )
with col_send:
    send_btn = st.button("Send ▶", use_container_width=True)

if (send_btn or user_input) and user_input.strip():
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                if st.session_state.mode == "document":
                    if not st.session_state.vectorstore:
                        response = "Please upload a PDF document first using the sidebar."
                    else:
                        response = chat_with_doc(user_input, st.session_state.vectorstore, api_key)
                else:
                    response = chat_general(user_input, st.session_state.messages[:-1], api_key)
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

st.markdown("""<div style="text-align:center;padding:16px;color:#475569;font-size:12px;
border-top:1px solid #1e293b;margin-top:20px">
Built by <strong style="color:#38bdf8">Chukka Akruthi Goud</strong> ·
LangChain · OpenAI · FAISS · Conversation Memory · Streamlit</div>""", unsafe_allow_html=True)
PYEOF
python3 -c "
import ast
with open('/mnt/user-data/outputs/project5_ai_chatbot.py') as f:
    source = f.read()
ast.parse(source)
print('Project 5: SYNTAX OK')
"
