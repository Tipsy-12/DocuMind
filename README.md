#  DocuMind - Multimodal Question-answering Document

This tool uses a multimodal RAG pipeline to summarize and extract insights from academic PDFs — including text, tables, and images. Powered by LangChain, Unstructured, Groq, Gemini 1.5 Flash, and DocArray, it enables intelligent understanding of research papers in one go.

  

---

###  Tech Stack

| Component         | Tool / Library                          |
|-------------------|------------------------------------------|
| LLM               | Gemini 1.5 Flash via LangChain           |
| PDF Processing    | Unstructured + Poppler      |
| OCR               | Tesseract (used within Unstructured)     |
| Embeddings        | GoogleGenerativeAIEmbeddings             |
| Vector Search     | DocArrayInMemorySearch                   |
| Text Splitting    | chunking_strategy="by_title" (Unstructured) |
| Image Captioning  | Gemini multimodal prompt via base64      |
| Text and Table Summarization     | LangChain PromptTemplate + Groq        |


---

### ⚙️ How to Use (Colab)
1. Open the notebook MMRAG.ipynb in Google Colab
2. Upload the research paper PDF
3. Add your Gemini API key
4. Enter your query in the space provided   
5. Run all cells in order.

---

### Setup Instructions (Local)
```bash

# Clone the repository
git clone https://github.com/Tipsy-12/Intelligent-Paper-Assistant.git
cd Intelligent-Paper-Assistant

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_TRACING_V2=false
```

---

### ⚙️ System Dependencies
You’ll need the following tools installed on your system:

📦 Linux / Ubuntu / Codespaces:
```bash
sudo apt update && sudo apt install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1
```
poppler-utils: for pdf2image to convert PDFs to images.

tesseract-ocr: for OCR capabilities in unstructured.

libgl1: required by opencv-python.

---





