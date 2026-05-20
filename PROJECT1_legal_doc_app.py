import os
import io
import math
import streamlit as st
import pypdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="LegalAI — Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; }
.main-header {
    background: linear-gradient(135deg, #1a1f3a 0%, #0f1628 100%);
    padding: 30px 40px; border-radius: 16px;
    border: 1px solid #2a3050; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.main-title { color: #60a5fa; font-size: 32px; font-weight: 700; margin: 0; }
.main-sub { color: #94a3b8; font-size: 15px; margin: 6px 0 0 0; }
.metric-card {
    background: #1a1f3a; border: 1px solid #2a3050;
    border-radius: 12px; padding: 20px; text-align: center;
}
.metric-val { color: #60a5fa; font-size: 28px; font-weight: 700; }
.metric-label { color: #94a3b8; font-size: 13px; margin-top: 4px; }
.result-box {
    background: #1a1f3a; border: 1px solid #2a3050;
    border-radius: 12px; padding: 24px; margin-top: 16px;
    line-height: 1.8; color: #e2e8f0;
}
.answer-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2f4a 100%);
    border: 1px solid #3b6ea5; border-radius: 12px; padding: 24px;
    margin-top: 16px; color: #e2e8f0; line-height: 1.8;
}
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin: 3px;
}
.badge-blue { background: #1e3a5f; color: #60a5fa; border: 1px solid #3b6ea5; }
.badge-green { background: #1a3a2a; color: #4ade80; border: 1px solid #2d6a4f; }
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 10px 24px !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ── Pure Python RAG Engine ───────────────────────────────────
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=150):
    """Pure Python character splitter to avoid text-splitter binary locks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def cosine_similarity(v1, v2):
    """Calculates similarity score manually using native math algorithms."""
    dot_prod = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if not mag1 or not mag2:
        return 0
    return dot_prod / (mag1 * mag2)

@st.cache_resource(show_spinner=False)
def process_and_vectorize_text(text, api_key):
    """Generates embeddings using the raw OpenAI client wrapper."""
    client = OpenAI(api_key=api_key)
    chunks = split_text_into_chunks(text)
    
    # Get embeddings for all chunks natively
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    
    vector_database = []
    for i, data in enumerate(response.data):
        vector_database.append({
            "text": chunks[i],
            "embedding": data.embedding
        })
    return vector_database

def search_similar_chunks(vector_database, query, api_key, k=4):
    """Finds top context blocks using our clean pure-math similarity engine."""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    scored_chunks = []
    for item in vector_database:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append((score, item["text"]))
        
    # Sort descending by score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:k]]

# ── Helper Processing Core ───────────────────────────────────
def extract_text_from_pdf(file_bytes):
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def summarize_doc(text, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "value": "You are a precise legal document assistant."},
            {"role": "user", "content": f"""Summarize this legal document clearly in plain English.
Include: parties involved, key obligations, important dates, penalty clauses, and key terms.
Use bullet points. Be concise and professional.

Document: {text[:12000]}

Summary:"""}
        ]
    )
    return response.choices[0].message.content

def answer_question(context_chunks, question, api_key):
    client = OpenAI(api_key=api_key)
    context_text = "\n\n---NEW CHUNK---\n\n".join(context_chunks)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise legal document assistant."},
            {"role": "user", "content": f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say "This information is not found in the document."
Be accurate, concise, and professional.

Context:
{context_text}

Question: {question}

Answer:"""}
        ]
    )
    return response.choices[0].message.content

def extract_clauses(text, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise legal document assistant."},
            {"role": "user", "content": f"""Extract key information from this legal document.
Return in this exact format:

PARTIES: [list parties]
EFFECTIVE DATE: [date]
TERMINATION DATE: [date or N/A]
KEY OBLIGATIONS: [bullet points]
PENALTY CLAUSES: [bullet points or None]
GOVERNING LAW: [jurisdiction]
CONFIDENTIALITY: [Yes/No and brief details]
DISPUTE RESOLUTION: [method]

Document: {text[:6000]}"""}
        ]
    )
    return response.choices[0].message.content

# ── Main UI Layout ───────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <p class="main-title">⚖️ LegalAI — Intelligent Document Analyzer</p>
  <p class="main-sub">LLM Pipeline · OpenAI GPT-3.5 · Pure Python Vector Matcher</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""),
                             help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.divider()
    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader("Upload Legal PDF", type="pdf")
    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    for badge in ["LLM Pipeline", "OpenAI GPT-3.5", "Pure Python Vector Engine", "Streamlit", "Prompt Engineering"]:
        st.markdown(f'<span class="badge badge-blue">{badge}</span>', unsafe_allow_html=True)

if uploaded:
    if not os.environ.get("OPENAI_API_KEY"):
        st.info("ℹ️ Please enter your OpenAI API Key in the sidebar to process the document.")
    else:
        current_api_key = os.environ.get("OPENAI_API_KEY")
        with st.spinner("📖 Reading and indexing document layers..."):
            file_bytes = uploaded.read()
            doc_text = extract_text_from_pdf(file_bytes)
            vector_db = process_and_vectorize_text(doc_text, current_api_key)

        word_count = len(doc_text.split())
        page_count = len(pypdf.PdfReader(io.BytesIO(file_bytes)).pages)
        char_count = len(doc_text)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{page_count}</div>'
                        f'<div class="metric-label">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{word_count:,}</div>'
                        f'<div class="metric-label">Words</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{char_count:,}</div>'
                        f'<div class="metric-label">Characters</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "💬 Ask Questions", "🔍 Extract Clauses", "📄 Raw Text"])

        with tab1:
            st.markdown("#### 📝 AI Document Summary")
            if st.button("Generate Summary", key="sum_btn"):
                with st.spinner("Summarizing legal parameters..."):
                    summary = summarize_doc(doc_text, current_api_key)
                st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown("#### 💬 Ask Questions About Your Document")
            question = st.text_input("Your question:", placeholder="Who are the parties involved?")
            if st.button("Get Answer", key="qa_btn") and question:
                with st.spinner("Calculating matching nodes and generating response..."):
                    matched_chunks = search_similar_chunks(vector_db, question, current_api_key)
                    answer = answer_question(matched_chunks, question, current_api_key)
                st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br><br>{answer}</div>', unsafe_allow_html=True)
                with st.expander("📎 Source Text Blocks Natively Matched"):
                    for i, chunk in enumerate(matched_chunks, 1):
                        st.markdown(f"**Context Block {i}:** {chunk[:300]}...")

        with tab3:
            st.markdown("#### 🔍 Extract Key Legal Clauses")
            if st.button("Extract Clauses", key="clause_btn"):
                with st.spinner("Parsing structure patterns..."):
                    clauses = extract_clauses(doc_text, current_api_key)
                st.markdown(f'<div class="result-box"><pre style="color:#e2e8f0;font-family:Inter;white-space:pre-wrap">{clauses}</pre></div>', unsafe_allow_html=True)

        with tab4:
            st.markdown("#### 📄 Extracted Text")
            st.text_area("Raw document text:", doc_text[:5000] + "..." if len(doc_text) > 5000 else doc_text, height=400)
else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#64748b">
        <div style="font-size:64px;margin-bottom:16px">⚖️</div>
        <h3 style="color:#94a3b8;font-weight:600">Upload a Legal Document to Begin</h3>
        <p style="color:#64748b;margin-top:8px">Upload any PDF contract, agreement, or legal document<br>
        and get AI-powered summaries, Q&A, and clause extraction.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:20px;color:#475569;font-size:12px;margin-top:40px;border-top:1px solid #1e293b">
    Built by <strong style="color:#60a5fa">Chukka Akruthi Goud</strong> · LLM Pipeline · OpenAI · Streamlit
</div>
""", unsafe_allow_html=True)
