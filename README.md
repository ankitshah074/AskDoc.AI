# 🧠 GenAI Document Assistant

**GenAI Document Assistant** is a multilingual AI-powered application that allows users to upload documents (PDF, DOCX, or TXT) and ask questions in any major language. The assistant intelligently understands the document, retrieves relevant context, and answers questions using advanced language models and vector similarity search.

check it- https://ask-doc-ai.streamlit.app/

---

## 🌟 Features

✅ Supports **PDF**, **Word (.docx)**, and **Text (.txt)** files  
✅ **Multilingual Q&A** — ask questions in Hindi, English, French, Spanish, Bengali, etc.  
✅ Uses **sentence embeddings** + **vector search (FAISS)** for intelligent context retrieval  
✅ Powered by **Hugging Face Transformers** and **LangChain**  
✅ Built with **Streamlit** for fast, clean web app deployment  

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Embeddings**: `distiluse-base-multilingual-cased-v2` from `sentence-transformers`  
- **QA Model**: `distilbert-base-cased-distilled-squad` via Hugging Face Transformers  
- **Vector Search**: FAISS (Facebook AI Similarity Search)  
- **Text Processing**: LangChain’s text splitter for chunking long documents  

---

## 🧠 How It Works

1. **Upload a Document**  
   Upload `.pdf`, `.txt`, or `.docx` format. Text is extracted and chunked into small segments.

2. **Embedding & Indexing**  
   Each chunk is converted into a dense vector using a multilingual sentence transformer and stored in a FAISS index.

3. **Ask a Question**  
   The user submits a query in any language. The query is embedded, and the assistant finds the most relevant document chunks.

4. **Answer Generation**  
   A pre-trained model analyzes the context and returns the most accurate answer to the user.

## Example Use Cases
📚 Students: Ask questions about lecture notes or study material
🧑‍💼 Professionals: Extract summaries from business reports or whitepapers
👨‍⚖️ Legal: Query long contracts or case files
📊 Research: Analyze papers or data documentation


---

## 🚀 Deployment

### 📦 Local Deployment

```bash
git clone https://github.com/ankitshah074/genai-doc-assistant.git
cd genai-doc-assistant
pip install -r requirements.txt
streamlit run app.py
