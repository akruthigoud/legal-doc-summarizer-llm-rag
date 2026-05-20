import os
import io
import streamlit as st
import pypdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

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

# ── Helper Functions ─────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

@st.cache_resource(show_spinner=False)
def build_vectorstore(text_hash, text):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents([text])
    return FAISS.from_documents(chunks, embeddings)

def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                      openai_api_key=os.getenv("OPENAI_API_KEY"))

def summarize_doc(text):
    llm = get_llm()
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    prompt = PromptTemplate(input_variables=["text"], template="""
Summarize this legal document clearly in plain English.
Include: parties involved, key obligations, important dates, penalty clauses, and key terms.
Use bullet points. Be concise and professional.

Document: {text}

Summary:""")
    chain = load_summarize_chain(llm, chain_type="map_reduce",
                                 map_prompt=prompt, combine_prompt=prompt)
    return chain.invoke({"input_documents": chunks})["output_text"]

def answer_question(vectorstore, question):
    llm = get_llm()
    prompt = PromptTemplate(input_variables=["context", "question"], template="""
You are a precise legal document assistant.
Answer the question using ONLY the context provided.
If the answer is not in the context, say "This information is not found in the document."
Be accurate, concise, and professional.

Context: {context}

Question: {question}

Answer:""")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": prompt}, return_source_documents=True)
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]

def extract_clauses(text):
    llm = get_llm()
    response = llm.invoke(f"""Extract key information from this legal document.
Return in this exact format:

PARTIES: [list parties]
EFFECTIVE DATE: [date]
TERMINATION DATE: [date or N/A]
KEY OBLIGATIONS: [bullet points]
PENALTY CLAUSES: [bullet points or None]
GOVERNING LAW: [jurisdiction]
CONFIDENTIALITY: [Yes/No and brief details]
DISPUTE RESOLUTION: [method]

Document: {text[:4000]}""")
    return response.content

# ── Main App ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <p class="main-title">⚖️ LegalAI — Intelligent Document Analyzer</p>
  <p class="main-sub">LLM + RAG Pipeline · OpenAI GPT · LangChain · Vector Search</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""),
                             help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.divider()
    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader("Upload Legal PDF", type="pdf",
                                  help="Upload any legal document, contract, or agreement")
    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    for badge in ["LLM · RAG · LangChain", "OpenAI GPT-3.5", "FAISS Storage",
                  "Streamlit · Python", "Prompt Engineering"]:
        st.markdown(f'<span class="badge badge-blue">{badge}</span>', unsafe_allow_html=True)

# Process uploaded file
if uploaded:
    if not os.getenv("OPENAI_API_KEY"):
        st.info("ℹ️ Please enter your OpenAI API Key in the sidebar to process the document.")
    else:
        with st.spinner("📖 Reading and indexing document..."):
            file_bytes = uploaded.read()
            doc_text = extract_text_from_pdf(file_bytes)
            text_hash = hash(doc_text)
            vectorstore = build_vectorstore(text_hash, doc_text)

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
                with st.spinner("Summarizing document with AI..."):
                    summary = summarize_doc(doc_text)
                st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown("#### 💬 Ask Questions About Your Document")
            st.markdown("*Ask anything about the document — parties, dates, obligations, clauses...*")
            question = st.text_input("Your question:", placeholder="Who are the parties involved in this contract?")
            if st.button("Get Answer", key="qa_btn") and question:
                with st.spinner("Searching document and generating answer..."):
                    answer, sources = answer_question(vectorstore, question)
                st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br><br>{answer}</div>',
                            unsafe_allow_html=True)
                with st.expander("📎 Source Passages Used"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:** {doc.page_content[:300]}...")

            st.markdown("**Try these example questions:**")
            examples = ["Who are the parties in this agreement?",
                        "What are the payment terms?",
                        "What happens in case of breach of contract?",
                        "What is the governing law?"]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    if st.button(ex, key=f"ex_{i}"):
                        with st.spinner("Answering..."):
                            ans, _ = answer_question(vectorstore, ex)
                        st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown("#### 🔍 Extract Key Legal Clauses")
            if st.button("Extract Clauses", key="clause_btn"):
                with st.spinner("Extracting legal clauses..."):
                    clauses = extract_clauses(doc_text)
                st.markdown(f'<div class="result-box"><pre style="color:#e2e8f0;font-family:Inter;white-space:pre-wrap">{clauses}</pre></div>',
                            unsafe_allow_html=True)

        with tab4:
            st.markdown("#### 📄 Extracted Text")
            st.text_area("Raw document text:", doc_text[:5000] + "..." if len(doc_text) > 5000 else doc_text,
                         height=400)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#64748b">
        <div style="font-size:64px;margin-bottom:16px">⚖️</div>
        <h3 style="color:#94a3b8;font-weight:600">Upload a Legal Document to Begin</h3>
        <p style="color:#64748b;margin-top:8px">Upload any PDF contract, agreement, or legal document<br>
        and get AI-powered summaries, Q&A, and clause extraction.</p>
        <br>
        <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap">
            <span class="badge badge-blue">LLM + RAG</span>
            <span class="badge badge-green">OpenAI GPT</span>
            <span class="badge badge-blue">FAISS Storage</span>
            <span class="badge badge-green">LangChain</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:20px;color:#475569;font-size:12px;margin-top:40px;
            border-top:1px solid #1e293b">
    Built by <strong style="color:#60a5fa">Chukka Akruthi Goud</strong> ·
    LLM + RAG · LangChain · OpenAI · FAISS · Streamlit
</div>
""", unsafe_allow_html=True)
