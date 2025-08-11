import streamlit as st
import os
import base64
from backend import (
    initialize_models,
    process_pdf,
    summarize_content,
    caption_images,
    create_retriever,
    setup_qa_chain,
    display_base64_image_streamlit,
    generate_recommended_questions,
    generate_further_recommended_questions
)
import tempfile
from PIL import Image

st.set_page_config(layout="wide")
st.title("Multimodal PDF Question Answering")

# Sidebar for API keys and PDF upload
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
# Groq API key is hardcoded in the backend.py as per the original notebook, 
# but if it were to be user-provided, it would be here.
# groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state variables
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "paper_title" not in st.session_state:
    st.session_state.paper_title = ""
if "text_count" not in st.session_state:
    st.session_state.text_count = 0
if "table_count" not in st.session_state:
    st.session_state.table_count = 0
if "image_count" not in st.session_state:
    st.session_state.image_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = []

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("PDF Preview")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Display PDF using an iframe
        st.markdown(f"""<iframe src="data:application/pdf;base64,{base64.b64encode(open(pdf_path, 'rb').read()).decode('utf-8')}" width="100%" height="700px" type="application/pdf"></iframe>""", unsafe_allow_html=True)

        if st.session_state.qa_chain is None and gemini_api_key:
            with st.spinner("Processing PDF and setting up QA system..."):
                try:
                    # Initialize models
                    llm_groq, llm_gemini, embeddings = initialize_models(gemini_api_key)

                    # Process PDF
                    elements, texts, tables, images_b64, paper_title = process_pdf(pdf_path)
                    st.session_state.paper_title = paper_title
                    st.session_state.text_count = len(texts)
                    st.session_state.table_count = len(tables)
                    st.session_state.image_count = len(images_b64)

                    # Summarize content
                    text_summaries, table_summaries, table_htmls = summarize_content(llm_groq, texts, tables)

                    # Caption images
                    image_captions = caption_images(llm_gemini, images_b64)

                    # Create retriever
                    retriever = create_retriever(texts, text_summaries, tables, table_summaries, images_b64, image_captions, embeddings)

                    # Setup QA chain
                    st.session_state.qa_chain = setup_qa_chain(llm_groq, retriever)
                    st.success("PDF processed and QA system ready!")

                    # Generate initial recommended questions
                    st.session_state.recommended_questions = ["What is this paper about?", "Can you summarize the main findings?", "How many tables are there?", "Are there any images?"]

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    st.session_state.qa_chain = None
    else:
        st.info("Upload a PDF and enter your Gemini API key to start.")

with col2:
    st.header("Chat with PDF")

    if st.session_state.qa_chain:
        st.write(f"**Paper Title:** {st.session_state.paper_title}")
        st.write(f"**Text Chunks:** {st.session_state.text_count}")
        st.write(f"**Tables:** {st.session_state.table_count}")
        st.write(f"**Images:** {st.session_state.image_count}")

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "rendered_content" in message and message["rendered_content"]:
                    for item in message["rendered_content"]:
                        if item["type"] == "table":
                            st.markdown(item["content"], unsafe_allow_html=True)
                        elif item["type"] == "image":
                            st.image(display_base64_image_streamlit(item["content"]), use_column_width=True)

        # Recommended questions
        if st.session_state.recommended_questions:
            st.markdown("**Recommended Questions:**")
            cols = st.columns(len(st.session_state.recommended_questions))
            for i, q in enumerate(st.session_state.recommended_questions):
                if cols[i].button(q, key=f"rec_q_{i}"):
                    st.session_state.user_question_triggered = q
                    st.rerun()

        # Chat input
        user_question = st.chat_input("Ask a question about the PDF:", key="chat_input")

        if user_question or (hasattr(st.session_state, "user_question_triggered") and st.session_state.user_question_triggered):
            if user_question:
                current_question = user_question
            else:
                current_question = st.session_state.user_question_triggered
                del st.session_state.user_question_triggered # Clear after use

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": current_question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(current_question)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, rendered_content = st.session_state.qa_chain(current_question)
                        st.markdown(response)
                        
                        # Display rendered content (tables and images)
                        if rendered_content:
                            for item in rendered_content:
                                if item["type"] == "table":
                                    st.markdown(item["content"], unsafe_allow_html=True)
                                elif item["type"] == "image":
                                    st.image(display_base64_image_streamlit(item["content"]), use_column_width=True)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response, "rendered_content": rendered_content})
                        
                        # Generate further recommended questions
                        st.session_state.recommended_questions = generate_further_recommended_questions(current_question, response)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    else:
        st.info("Please upload a PDF and enter your API key(s) in the sidebar to enable chat.")




