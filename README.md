# RAG-Based PDF Explorer 

An interactive, Retrieval-Augmented Generation (RAG) web application built using **Streamlit**, **ChromaDB**, and **Google Gemini** (`google-generativeai`). This tool allows users to upload any PDF document and ask conversational questions directly against its contents with accurate, context-aware answers.

---

##  Features
* **PDF Document Ingestion:** Upload multi-page PDF documents effortlessly via a clean drag-and-drop web dashboard.
* **Vectorized Search Infrastructure:** Automatically splits document text chunks and indexes them into a highly efficient local instance of **ChromaDB**.
* **AI-Powered Question Answering:** Leverages advanced contextual grounding via **Google Gemini models** to answer complex queries without leaking private text or background data.
* **API Key Safety:** Built-in validation settings directly inside the Streamlit sidebar layout to ensure secure, credential-safe testing.

---

##  Tech Stack
* **Frontend Dashboard:** [Streamlit](https://streamlit.io)
* **Vector Database:** [ChromaDB](https://trychroma.com)
* **Orchestration & Splitting:** [LangChain Text Splitters](https://langchain.com)
* **LLM Engine:** [Google Generative AI](https://pypi.org)
* **PDF Processing:** `pypdf`

---

##  Local Installation & Setup

Follow these simple steps to run this project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd RAG_based_PDF_Explorer
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure all application requirements are configured correctly:
```bash
pip install -r requirements.txt
```

### 4. Launch the Web App
Run the local Streamlit environment server directly from your terminal pane:
```bash
streamlit run main.py
```

---

##  Configuration & Environment Safety
To successfully process embeddings, ensure you input your specific API key into the secure credential field layout inside the web interface sidebar before uploading documents. 

*Note: Your `.venv`, `.env` configuration keys, and local test variations of your `chroma_db/` document text vectors are intentionally omitted from GitHub tracking records via version exclusions to preserve security.*
