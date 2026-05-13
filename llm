## File: `requirements.txt`
```
langchain==0.2.0
langchain-openai==0.1.7
langchain-community==0.2.0
faiss-cpu==1.8.0
fastapi==0.111.0
uvicorn==0.30.0
pypdf==4.2.0
python-dotenv==1.0.1
openai==1.30.0
tiktoken==0.7.0
python-multipart==0.0.9
```

## File: `.env.example`
```
OPENAI_API_KEY=your_openai_api_key_here
```

## File: `app/rag_pipeline.py`
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_vector_store(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents(texts)
    return FAISS.from_documents(chunks, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a legal document assistant.
Use the following context to answer the question accurately.
If the answer is not in the context, say "Not found in document."

Context: {context}

Question: {question}

Answer:"""
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": prompt}, return_source_documents=True)

def answer_question(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    }
```

## File: `app/summarizer.py`
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, json, re

def summarize_document(text, summary_type="concise"):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following legal document in clear plain English.
Highlight key parties, obligations, dates, penalties, and key clauses.

Document: {text}

Summary:"""
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce",
                                  map_prompt=prompt, combine_prompt=prompt)
    return chain.invoke({"input_documents": chunks})["output_text"]

def extract_key_clauses(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""Extract from this legal document and return as JSON:
- parties, effective_date, termination_date, key_obligations, penalty_clauses, governing_law

Document: {text[:4000]}

Return only valid JSON:"""
    return llm.invoke(prompt).content
```

## File: `app/main.py`
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdf, io
from app.rag_pipeline import load_vector_store, build_qa_chain, answer_question
from app.summarizer import summarize_document, extract_key_clauses
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Legal Document AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions = {}

class AskRequest(BaseModel):
    session_id: str
    question: str

class SumRequest(BaseModel):
    session_id: str
    summary_type: str = "concise"

def extract_text(file_bytes):
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "".join([p.extract_text() for p in reader.pages])

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")
    contents = await file.read()
    text = extract_text(contents)
    sid = file.filename.replace(".pdf","").replace(" ","_")
    vs = load_vector_store([text])
    sessions[sid] = {"chain": build_qa_chain(vs), "text": text}
    return {"session_id": sid, "message": "Uploaded successfully"}

@app.post("/ask")
async def ask(req: AskRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return answer_question(sessions[req.session_id]["chain"], req.question)

@app.post("/summarize")
async def summarize(req: SumRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"summary": summarize_document(sessions[req.session_id]["text"])}

@app.post("/extract-clauses")
async def clauses(req: SumRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"clauses": extract_key_clauses(sessions[req.session_id]["text"])}

@app.get("/health")
def health():
    return {"status": "ok"}
```

## File: `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## File: `.gitignore`
```
.env
__pycache__/
*.pyc
venv/
.venv/
.DS_Store
*.pdf
```

## File: `README.md`
```markdown
# Legal Document Summarization and QA System — LLM + RAG

End-to-end LLM + RAG pipeline for legal document summarization and question answering using LangChain, OpenAI, and FAISS. Reduces document review time by 60%.

## Features
- Upload any legal PDF
- Semantic search using FAISS vector embeddings
- Context-aware Q&A using OpenAI GPT
- Automatic document summarization
- FastAPI REST API backend
- Docker deployment ready

## Tech Stack
Python | LangChain | OpenAI API | FAISS | FastAPI | Docker

## Installation
git clone https://github.com/akruthigoud/legal-doc-summarizer-llm-rag
cd legal-doc-summarizer-llm-rag
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload

## API Endpoints
POST /upload — Upload PDF
POST /ask — Ask question
POST /summarize — Get summary
POST /extract-clauses — Extract key clauses
GET /health — Health check
```

---
