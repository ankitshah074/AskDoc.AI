import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile

# Page setup
st.set_page_config(page_title="GenAI Doc Assistant", layout="wide")
st.title("📄 GenAI Document Assistant 🌍")
st.markdown("""
Upload a **PDF**, **TXT**, or **DOCX** file.  
Ask questions in **Hindi, English, French, Spanish, Bengali, Arabic**, or any other major language.  
Powered by **M-BERT** & **Sentence Transformers**
""")

# Load models only once
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embedder, qa

embedder, qa_pipeline = load_models()

# Read DOCX
def extract_docx_text(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# Upload section
uploaded_file = st.file_uploader("📁 Upload your document", type=["pdf", "txt", "docx"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    # Read based on file type
    if ext == "pdf":
        reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "txt":
        text = uploaded_file.read().decode("utf-8")
    elif ext == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = extract_docx_text(tmp_path)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Text chunking
            # splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            # chunks = splitter.split_text(text)
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Embed & create FAISS index
    with st.spinner("🔍 Indexing document..."):
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

    # Input query
    query = st.text_input("🗣️ Ask your question (in any language):")

    if query:
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        top_chunks = [chunks[i] for i in I[0]]
        context = " ".join(top_chunks)

        with st.spinner("🧠 Thinking..."):
            result = qa_pipeline(question=query, context=context)
            st.success(f"✅ Answer: {result['answer']}")

        with st.expander("📚 Context Used"):
            st.write(context[:1200] + "...")
# Add footer or credit
st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 **Built by Ankit Shah**")
st.sidebar.markdown("[📎 LinkedIn](https://www.linkedin.com/in/ank-it-shah/) | [🐙 GitHub](https://github.com/ankitshah074)")
