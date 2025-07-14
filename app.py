# ==============================================================================
# 1. IMPORT CORE LIBRARIES
# ==============================================================================
# Import libraries that are essential for the basic UI and file handling.
# Heavy AI-related libraries will be imported later, only when needed.
import streamlit as st
from pypdf import PdfReader
import os

# ==============================================================================
# 2. APP CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Doc Q&A Agent (RAG)",
    page_icon="🧠",
    layout="wide"
)

# ==============================================================================
# 3. API KEY MANAGEMENT
# ==============================================================================
# This section checks for the API key at the start.
# The app will show an error and stop if the key is not found.
try:
    # Set the environment variable for other libraries to use.
    os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # If the key is not found in Streamlit's secrets, display an error and halt.
    st.error("Gemini API key not found. Please add it to your .streamlit/secrets.toml file.", icon="🚨")
    st.stop() # This command stops the script execution immediately.


# ==============================================================================
# 4. HELPER FUNCTIONS (with Lazy Loading)
# ==============================================================================
# These functions now import their required libraries internally.
# This prevents the app from crashing on startup if a library has an issue.

def get_text_from_files(uploaded_files):
    """Reads and extracts text from a list of uploaded files."""
    text = ""
    for uploaded_file in uploaded_files:
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension.lower() == ".pdf":
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error reading PDF {uploaded_file.name}: {e}", icon="📄")
        elif file_extension.lower() == ".txt":
            text += uploaded_file.read().decode("utf-8")
    return text

def get_text_chunks(text):
    """Splits text into smaller chunks."""
    # LAZY IMPORT: Import the library only when this function is called.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    # LAZY IMPORT: Import necessary libraries here.
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.vectorstores import FAISS
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        #st.error(f"Error creating vector store: {e}", icon="�")
        st.error(f"Error creating vector store: {e}", icon="🚨") 
        return None

def get_conversational_chain():
    """Creates the question-answering chain."""
    # LAZY IMPORT: Import necessary libraries here.
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain

    prompt_template = """
    You are a helpful AI assistant. Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# ==============================================================================
# 5. MAIN APP FUNCTION
# ==============================================================================
# Encapsulating the app logic in a main function is good practice.

def main():
    """The main function that runs the Streamlit application."""
    st.title("📄🧠 AI Document Q&A Agent with RAG")
    st.markdown("---")
    st.markdown("""
    This advanced agent uses **Retrieval-Augmented Generation (RAG)** to answer your questions.
    1.  **Upload one or more documents** (`.pdf` or `.txt`).
    2.  The system will create a searchable **Vector Database** from the content.
    3.  **Ask a question**, and the AI will find relevant information to give you a precise answer.
    """)

    # Initialize session state variables if they don't exist.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # UI Layout
    col1, col2 = st.columns([1, 1])

    # Left Column: Document Upload and Processing
    with col1:
        st.header("1. Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )

        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    raw_text = get_text_from_files(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        if st.session_state.vector_store:
                            st.success("Documents processed successfully!", icon="✅")
                    else:
                        st.warning("Could not extract text from the uploaded file(s).", icon="⚠️")
            else:
                st.warning("Please upload at least one document.", icon="📁")

    # Right Column: Q&A and Chat History
    with col2:
        st.header("2. Ask Questions")
        
        if st.session_state.vector_store:
            user_question = st.text_input("Enter your question here:", key="question_input")

            if user_question:
                with st.spinner("Finding answer..."):
                    try:
                        docs = st.session_state.vector_store.similarity_search(user_question)
                        chain = get_conversational_chain()
                        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                        st.session_state.chat_history.append(("You", user_question))
                        st.session_state.chat_history.append(("Bot", response["output_text"]))
                    except Exception as e:
                        st.error(f"An error occurred: {e}", icon="🔥")
                        

            if st.session_state.chat_history:
                st.subheader("Conversation History")
                for role, text in reversed(st.session_state.chat_history):
                    if role == "You":
                        st.markdown(f'<div style="text-align: right; margin-bottom: 10px;"><div style="display: inline-block; padding: 10px; border-radius: 10px; background-color: #dcf8c6; color: #000000; max-width: 80%;"><b>You:</b> {text}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="text-align: left; margin-bottom: 10px;"><div style="display: inline-block; padding: 10px; border-radius: 10px; background-color: #f1f0f0; color: #000000; max-width: 80%;"><b>Bot:</b> {text}</div></div>', unsafe_allow_html=True)
        else:
            st.info("Please upload and process documents to start the conversation.")


# ==============================================================================
# 6. SCRIPT EXECUTION
# ==============================================================================
# This ensures that the main function is called only when the script is run directly.
if __name__ == "__main__":
    main()
