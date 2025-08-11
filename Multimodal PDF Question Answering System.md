# Multimodal PDF Question Answering System

A Streamlit-based web application that allows users to upload PDF documents and ask questions about their content. The system can handle text, tables, and images within PDFs, providing comprehensive answers with relevant context.

## Features

- **PDF Upload and Preview**: Upload PDF files and preview them directly in the browser
- **Multimodal Processing**: Extracts and processes text, tables, and images from PDFs
- **Intelligent Question Answering**: Uses advanced AI models to answer questions about the PDF content
- **Context-Aware Responses**: Shows relevant tables and images when answering questions
- **Chat Interface**: Conversational interface with chat history
- **Recommended Questions**: Provides suggested questions to help users explore the document
- **Basic Document Statistics**: Shows paper title, number of text chunks, tables, and images

## Requirements

- Python 3.11+
- Gemini API Key (for embeddings and image captioning)
- Groq API Key (hardcoded in the backend for text processing)

## Installation

1. Clone or download the project files:
   ```bash
   git clone <repository-url>
   cd pdf_qa_app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install system dependencies (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install -y poppler-utils ghostscript tesseract-ocr
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the displayed URL (typically `http://localhost:8501`)

3. In the sidebar:
   - Enter your Gemini API Key
   - Upload a PDF file using the file uploader

4. Once the PDF is processed:
   - View the PDF preview on the left side
   - See basic document statistics (title, text chunks, tables, images)
   - Use the chat interface on the right to ask questions
   - Click on recommended questions for quick exploration

## API Keys

### Gemini API Key
- Required for embeddings and image captioning
- Enter this in the sidebar when using the application
- Get your key from: https://makersuite.google.com/app/apikey

### Groq API Key
- Used for text summarization and question answering
- Currently hardcoded in `backend.py` (line 19)
- Get your key from: https://console.groq.com/

## File Structure

```
pdf_qa_app/
├── app.py              # Main Streamlit application
├── backend.py          # Backend processing functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

1. **PDF Processing**: Uses the `unstructured` library to extract text, tables, and images from PDFs
2. **Content Summarization**: Summarizes text chunks and tables using Groq's LLaMA model
3. **Image Captioning**: Generates descriptions for images using Google's Gemini model
4. **Vector Storage**: Creates embeddings and stores them in ChromaDB for similarity search
5. **Question Answering**: Retrieves relevant context and generates answers using the LLaMA model

## Features in Detail

### PDF Preview
- Displays the uploaded PDF directly in the browser using an embedded iframe
- Allows users to reference the original document while asking questions

### Multimodal Content Handling
- **Text**: Extracts and summarizes text chunks for better context understanding
- **Tables**: Preserves table structure and renders them in HTML format when relevant
- **Images**: Generates detailed captions and displays images when they're relevant to questions

### Chat Interface
- Maintains conversation history
- Supports follow-up questions
- Displays relevant tables and images inline with responses

### Recommended Questions
- Provides initial suggested questions when a PDF is first loaded
- Updates recommendations based on the conversation context
- Helps users discover interesting aspects of the document

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed correctly
2. **PDF Processing Errors**: Ensure the PDF is not corrupted and contains readable content
3. **API Key Errors**: Verify that your Gemini API key is valid and has sufficient quota
4. **Memory Issues**: Large PDFs may require more memory; consider using smaller files for testing

### Performance Tips

- The system includes rate limiting to avoid API quota issues
- Processing time depends on PDF size and complexity
- Image captioning may take longer for PDFs with many images

## Limitations

- Currently optimized for research papers and academic documents
- Processing time increases with PDF size and complexity
- Requires internet connection for API calls
- Some PDF formats may not be fully supported

## Development

To modify or extend the application:

1. **Backend Logic**: Edit `backend.py` to change processing logic
2. **Frontend Interface**: Modify `app.py` to update the Streamlit interface
3. **Dependencies**: Update `requirements.txt` for new packages

## License

This project is provided as-is for educational and research purposes.

