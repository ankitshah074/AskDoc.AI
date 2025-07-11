import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings
import numpy as np
import tempfile
import os
import pickle

# Page setup
st.set_page_config(page_title="GenAI Doc Assistant", layout="wide")
st.title("📄 GenAI Document Assistant 🌍")
st.markdown("""
Upload a **PDF**, **TXT**, or **DOCX** file.  
Ask questions in **Hindi, English, French, Spanish, Bengali, Arabic**, or any other major language.  
Powered by **M-BERT** & **Sentence Transformers**
""")
from dotenv import load_dotenv
load_dotenv()

def extract_text_from_page(page):
    """Extract text from a page with orientation correction using OCR if needed."""
    text = page.extract_text()
    return text
def ocr_extract_text(pdf):
    """Extract text from an image-based or rotated PDF using OCR."""
    images = convert_from_path(uploaded_file)
    text = ""
    
    for image in images:
        # Use OCR to detect text in the correct orientation
        text += pytesseract.image_to_string(image)
    
    return text
# Read DOCX
def extract_docx_text(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
def main():
# Upload section
    uploaded_file = st.file_uploader("📁 Upload your document", type=["pdf", "txt", "docx"])

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()

        # Read based on file type
        if ext == "pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = extract_text_from_page(page)
                if page_text:
                    text += page_text
                else:
                # If no text was extracted from any pages, use OCR on the entire PDF
                    text = ocr_extract_text(uploaded_file)
                    break
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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text=text)
        if chunks:
                store_name = uploaded_file.name[:-4]

                if os.path.exists(f"{store_name}_chunks.pkl"):
                    # Load stored chunks and recreate vector store
                    with open(f"{store_name}_chunks.pkl", 'rb') as f:
                        chunks = pickle.load(f)
                    
                    with st.spinner("🔍 Indexing document..."):
                        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        vector_store = Chroma.from_texts(chunks, embedding=embeddings)
                else:
                    with st.spinner("Downloading and loading embeddings, please wait..."):
                        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    
                    st.success("Embeddings loaded successfully!")
                    
                    vector_store = Chroma.from_texts(chunks, embedding=embeddings)

                    # Store the chunks for future use
                    with open(f"{store_name}_chunks.pkl", "wb") as f:
                        pickle.dump(chunks, f)
        # Embed & create FAISS index
        # with st.spinner("🔍 Indexing document..."):
        #     embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        #     index = Chroma.from_texts(texts=chunks, embedding=embeddings)

        # Input query
        query = st.text_input("🗣️ Ask your question (in any language):")

        if query:
            # query_embedding = embedder.encode([query])
            # D, I = index.search(np.array(query_embedding), k=3)
            # top_chunks = [chunks[i] for i in I[0]]
            # context = " ".join(top_chunks)
            docs = vector_store.similarity_search(query=query, k=5)
            llm = ChatGroq(model="llama3-8b-8192")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            st.write("🤖 **Response:**", response)

if __name__ == "__main__":
    main()
# Add footer or credit
st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 **Built by Ankit Shah**")
st.sidebar.markdown("[📎 LinkedIn](https://www.linkedin.com/in/ank-it-shah/) | [🐙 GitHub](https://github.com/ankitshah074)")
