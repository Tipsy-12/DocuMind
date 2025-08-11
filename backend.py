import os
import base64
import time
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from PIL import Image
import io

# Set environment variables (replace with actual keys or use Streamlit secrets)
# Hardcoded GROQ API key as per original notebook
os.environ["GROQ_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize models
def initialize_models(google_api_key):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm_groq, llm_gemini, embeddings

# PDF Processing
def process_pdf(file_path):
    # First partition for chunked text and tables
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=False,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts, tables = [], []
    paper_title = None

    for chunk in chunks:
        # Capture the first title if not already set
        if paper_title is None and chunk.category == "Title":
            paper_title = chunk.text.strip()

        # Separate tables and text
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    # Fallback if title not found
    if not paper_title:
        paper_title = "Unknown Paper Title"

    # Extract base64 images from CompositeElement chunks
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)

    # Second partition for table structure
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        model_name="yolox"
    )

    # Keep original Table elements directly
    tables_only = []
    for el in elements:
        if el.category == "Table":  # category is more reliable than type string
            tables_only.append(el)  # store the element itself

    return texts, tables_only, images_b64, paper_title

# Summarization
def summarize_content(llm_groq, texts, tables_only):
    summary_prompt = ChatPromptTemplate.from_template("""
You are an assistant tasked with summarizing tables and text.
Give a clear and thorough summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message with "Here is a summary".

Table or text chunk: {element}
""")

    summarize_chain = {"element": lambda x: x} | summary_prompt | llm_groq | StrOutputParser()

    text_summaries = summarize_in_batches(summarize_chain, texts, batch_size=2, delay=10)
    table_summaries = summarize_in_batches(summarize_chain, tables_only, batch_size=2, delay=10)
    
    # Extract table HTML content
    table_htmls = [
        getattr(t.metadata, "text_as_html", "") for t in tables_only
    ]
    
    return text_summaries, table_summaries, table_htmls

def summarize_in_batches(summarize_chain, content_list, batch_size=2, delay=10):
    summaries = []
    for i in range(0, len(content_list), batch_size):
        batch = content_list[i:i+batch_size]
        result = summarize_chain.batch(batch, {"max_concurrency": batch_size})
        summaries.extend(result)
        if i + batch_size < len(content_list):
            print(f"Sleeping {delay}s to avoid rate limits...")
            time.sleep(delay)
    return summaries

# Image Captioning
def caption_images(llm_gemini, images_b64):
    prompt_template = """You are a research assistant. Describe the image provided in detail."""
    
    image_captions = []
    for img_b64 in images_b64:
        try:
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ])
            caption = llm_gemini.invoke([message]).content
            image_captions.append(caption)
        except Exception as e:
            print(f"Error captioning image: {e}")
            image_captions.append("Error generating caption")
    return image_captions

def create_retriever(texts, text_summaries, tables_only, table_summaries, images_b64, image_captions, embeddings):
    # Initialize the Docstore (in-memory for this example)
    store = InMemoryStore()

    # Initialize the Vectorstore (ChromaDB in-memory for this example)
    vectorstore = Chroma(collection_name="multi-modal-rag", embedding_function=embeddings)

    # Initialize the MultiVectorRetriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )

    # Add texts - store as Document objects
    for i, text_chunk in enumerate(texts):
        doc_id = f"text_{i}"
        # Create Document object for text
        text_doc = Document(
            page_content=text_chunk.text,
            metadata={"type": "text", "doc_id": doc_id}
        )
        retriever.docstore.mset([(doc_id, text_doc)])
        vectorstore.add_texts([text_summaries[i]], metadatas=[{"doc_id": doc_id, "type": "text_summary"}])

    # Add tables - store as Document objects with HTML content
    for i, table_chunk in enumerate(tables_only):
        doc_id = f"table_{i}"
        # Create Document object for table with HTML content
        table_html = getattr(table_chunk.metadata, "text_as_html", "")
        table_doc = Document(
            page_content=table_html,
            metadata={"type": "table", "doc_id": doc_id}
        )
        retriever.docstore.mset([(doc_id, table_doc)])
        vectorstore.add_texts([table_summaries[i]], metadatas=[{"doc_id": doc_id, "type": "table_summary"}])

    # Add images - store as Document objects with base64 content
    for i, img_b64 in enumerate(images_b64):
        doc_id = f"image_{i}"
        # Create Document object for image
        image_doc = Document(
            page_content=img_b64,
            metadata={"type": "image", "doc_id": doc_id}
        )
        retriever.docstore.mset([(doc_id, image_doc)])
        vectorstore.add_texts([image_captions[i]], metadatas=[{"doc_id": doc_id, "type": "image_caption"}])

    return retriever

# Question Answering Chain
def setup_qa_chain(llm_groq, retriever):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep your answer concise and to the point."),
        ("user", "Question: {question}\\nContext: {context}")
    ])
    qa_chain = qa_prompt | llm_groq | StrOutputParser()

    def answer_question(question):
        docs = retriever.get_relevant_documents(question)
        context_text = ""
        rendered_content = []
        
        for doc in docs:
            doc_type = doc.metadata.get("type", None)
            
            if doc_type == "text_summary":
                # For text summaries, use the summary as context
                context_text += doc.page_content + "\\n"
                rendered_content.append({"type": "text", "content": doc.page_content})
                
            elif doc_type == "table_summary":
                # Retrieve the original table document from the docstore
                original_table_doc = retriever.docstore.mget([doc.metadata["doc_id"]])[0]
                # Use the table HTML content for context
                context_text += original_table_doc.page_content + "\\n"
                rendered_content.append({"type": "table", "content": original_table_doc.page_content})
                
            elif doc_type == "image_caption":
                # Retrieve the original image document from the docstore
                original_image_doc = retriever.docstore.mget([doc.metadata["doc_id"]])[0]
                # Use the image caption for context
                context_text += doc.page_content + "\\n"
                rendered_content.append({"type": "image", "content": original_image_doc.page_content})

        response = qa_chain.invoke({"question": question, "context": context_text})
        return response, rendered_content

    return answer_question

# Helper to display base64 images (for Streamlit)
def display_base64_image_streamlit(base64_code):
    return Image.open(io.BytesIO(base64.b64decode(base64_code)))

# Function to generate recommended questions
def generate_recommended_questions(last_question=None, last_answer=None):
    if last_question is None:
        # Initial recommended questions
        return [
            "What is this paper about?",
            "Can you summarize the main findings?",
            "How many tables are there?",
            "Are there any images?"
        ]
    else:
        # Further recommended questions based on conversation
        return [
            f"Tell me more about {last_question.split(' ')[-1]}",
            "What are the key findings?",
            "Summarize the methodology."
        ]

# Function to generate further recommended questions
def generate_further_recommended_questions(last_question, last_answer):
    return [
        "What are the limitations of this study?",
        "What future work is suggested?",
        "How does this compare to other research?"
    ]

