import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
import tiktoken
import numpy as np

load_dotenv()

st.set_page_config(page_title="RAG Q&A", page_icon="📄", layout="wide")

@st.cache_resource
def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = get_client()


def read_pdf(file):
    parts = []
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        parts.append(page.extract_text() or '')
    return "\n\n".join(parts)


def chunk(text, size=1000, overlap=200):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + size]
        chunks.append(enc.decode(chunk_tokens))
        i += size - overlap
    return chunks


def embed(text):
    text = text.replace("\n", " ").strip()
    res = client.embeddings.create(model="text-embedding-3-small", input=text)
    return res.data[0].embedding


def embed_all(chunks):
    embeddings = []
    bar = st.progress(0)
    for i, c in enumerate(chunks):
        embeddings.append(embed(c))
        bar.progress((i + 1) / len(chunks))
    bar.empty()
    return embeddings


def similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_best(question, chunks, embeddings, n=3):
    q_emb = embed(question)
    scores = [(i, similarity(q_emb, emb)) for i, emb in enumerate(embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scores[:n]]


st.title("📄 Document Q&A")
st.markdown("Upload a document and ask questions")

with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
    if file:
        st.success(f"✓ {file.name}")

if not file:
    st.info("← Upload a document to start")
else:
    if 'ready' not in st.session_state:
        with st.spinner("Processing..."):
            if file.name.endswith('.pdf'):
                content = read_pdf(file)
            else:
                content = file.read().decode('utf-8')
            
            chunks = chunk(content)
            st.info(f"Embedding {len(chunks)} chunks...")
            embeddings = embed_all(chunks)
            
            st.session_state.content = content
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.ready = True
            st.session_state.history = []
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", f"{len(st.session_state.content):,}")
    c2.metric("Chunks", len(st.session_state.chunks))
    c3.metric("Questions", len(st.session_state.history))
    
    st.divider()
    st.subheader("💬 Ask Questions")
    
    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
    
    with st.form(key="qform", clear_on_submit=True):
        question = st.text_input("Your question:")
        submit = st.form_submit_button("Ask")
    
    if submit and question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant = find_best(question, st.session_state.chunks, st.session_state.embeddings)
                context = "\n\n".join(relevant)
                
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer from context only."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}
                    ],
                    temperature=0.3
                )
                
                ans = res.choices[0].message.content
                st.write(ans)
        
        st.session_state.history.append((question, ans))
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()