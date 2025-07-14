AI Document Q&A Agent with RAG
This project implements an AI-powered Document Question-Answering Agent using Retrieval-Augmented Generation (RAG). It allows users to upload PDF and TXT documents, build a searchable knowledge base from their content, and then ask questions to get precise answers based only on the provided documents.

✨ Features
Document Upload: Supports uploading multiple PDF (.pdf) and text (.txt) files.

Text Extraction: Extracts raw text content from uploaded documents.

Intelligent Chunking: Splits large documents into smaller, overlapping chunks for optimized processing.

Semantic Search: Converts text chunks into vector embeddings and stores them in a FAISS vector database for efficient similarity search.

Retrieval-Augmented Generation (RAG): Combines document retrieval with a Large Language Model (Google Gemini 1.5 Flash) to generate accurate, context-aware answers.

Conversational Interface: Provides a user-friendly chat interface built with Streamlit to interact with the agent.

Contextual Answers: Answers questions strictly based on the content of the uploaded documents, reducing hallucinations.

Real-time Feedback: Displays processing status, success messages, and error alerts.

🏗️ System Architecture: The "Smart Librarian" Approach
Imagine you have a massive library filled with documents, and you want to quickly find answers to your questions. A traditional librarian might struggle. Your AI agent is like a "Smart Librarian" that combines two key abilities:

Retrieval (Finding Relevant Books): When you ask a question, this librarian doesn't read every book. Instead, it quickly scans the index of all books, identifies the most relevant chapters or pages, and pulls only those specific parts.

Generation (Summarizing and Answering): Once the relevant parts are found, the librarian reads only those parts carefully and then synthesizes a clear, concise answer to your question, making sure to stick to the information in the retrieved sections.

This combination of Retrieval and Generation is precisely what Retrieval-Augmented Generation (RAG) means. It makes the AI more efficient, more accurate, and less prone to "making things up" (hallucinating) because it's always grounded in the documents you provide.

Architecture Overview
The core components and their interactions are as follows:

User Interface (Streamlit): Handles file uploads, question input, and displays answers/chat history.

Document Ingestion & Processing: Extracts text, splits into chunks, and generates embeddings.

Vector Database (FAISS): Stores text embeddings for efficient semantic retrieval.

Generative Model (Google Gemini 1.5 Flash): The Large Language Model (LLM) that generates answers.

Conversational Chain (LangChain): Orchestrates the retrieval and generation process.

Workflow Diagram
graph TD
    subgraph User Interaction
        A[Start: User Opens App] --> B{Upload Files (PDF/TXT)};
        B --> C{Click "Process Documents"};
    end

    subgraph Document Processing & Indexing
        C --> D[Extract Text from Files];
        D --> E[Split Text into Chunks];
        E --> F[Generate Embeddings for Chunks];
        F --> G[Build FAISS Vector Store];
        G --&gt; H[Processing Complete: Ready for Questions];
    end

    subgraph Question Answering (RAG Flow)
        H --> I{User Enters Question};
        I --> J[Generate Embedding for Question];
        J --> K[Search FAISS Vector Store for Relevant Chunks];
        K --> L[Retrieve Top-K Most Similar Chunks (Context)];
        L --> M{Prepare Prompt with Context & Question};
        M --> N[Send Prompt to LLM (Gemini 1.5 Flash)];
        N --> O[LLM Generates Answer];
        O --> P[Display Answer to User];
        P --> Q{Add to Chat History};
        Q --&gt; I;
    end

    subgraph Error Handling
        D --&gt;|Error reading file| R[Display Error to User];
        F --&gt;|Error creating embeddings| R;
        J --&gt;|Error searching vector store| R;
        N --&gt;|Error with LLM response| R;
    end

💡 Key Design Decisions: Why We Built It This Way
Every part of this system was chosen for a specific reason to make it effective and user-friendly.

Streamlit for the User Interface (UI):

Decision: Chosen for its ease and speed in building interactive web applications purely with Python.

Benefit: Enables rapid prototyping and deployment, allowing users to easily interact without complex web development.

Document Pre-processing for Smart Searching:

Decision: Text is extracted (pypdf) and then chopped into smaller, overlapping "chunks" (RecursiveCharacterTextSplitter).

Benefit: Manages LLM context window limits and improves retrieval relevance by focusing on smaller, more specific text segments.

Vector Embeddings for "Understanding" Meaning:

Decision: Each text chunk is converted into a numerical vector (embedding) using GoogleGenerativeAIEmbeddings.

Benefit: Captures the semantic meaning of text, enabling semantic search to find relevant information even if exact keywords aren't used.

FAISS for Fast Information Retrieval:

Decision: Embeddings are stored in a FAISS vector store.

Benefit: FAISS is highly optimized for blazing-fast similarity searches, crucial for quickly finding the most relevant document chunks.

Gemini 1.5 Flash for Efficient Generation:

Decision: Utilizes ChatGoogleGenerativeAI with the gemini-1.5-flash-latest model.

Benefit: Gemini Flash is Google's fastest and most cost-effective Gemini model, ideal for low-latency, high-volume Q&A. A temperature of 0.3 ensures focused and factual answers.

LangChain for Orchestration:

Decision: LangChain is used to tie all components together into a seamless "chain" of operations.

Benefit: Provides standardized interfaces, making it easy to integrate and potentially swap out different LLMs, embedding models, or vector stores in the future.

Prompt Engineering for Guided Answers:

Decision: A PromptTemplate explicitly guides the LLM on how to use the provided context.

Benefit: Crucial for preventing hallucinations; the LLM is instructed to state if the answer is not in the context, ensuring answers are grounded in the documents.

Streamlit Session State for Memory:

Decision: vector_store and chat_history are stored in st.session_state.

Benefit: Maintains application state across Streamlit's reruns, ensuring processed documents and conversation history persist, providing a continuous user experience.

Lazy Loading for Better Startup:

Decision: Heavy AI libraries are imported inside the functions that use them.

Benefit: Allows the Streamlit app to start faster and display initial UI elements more gracefully, even if there are issues with specific AI library imports.

Robust Error Handling for User Experience:

Decision: try-except blocks are placed around critical operations, displaying user-friendly messages with st.error().

Benefit: Provides clear, actionable feedback to the user in case of errors (e.g., file reading issues, API key problems), preventing app crashes and improving usability.

🚶‍♂️ Code Walkthrough
The core logic of the application resides in app.py. Below is a breakdown of its main sections and key functions.

app.py
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
    from langchain_community.vectorstores import FAISS
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
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
if __name__ == "__main__":
    main()

1. Initial Setup and Configuration
# ==============================================================================
# 1. IMPORT CORE LIBRARIES
# ==============================================================================
import streamlit as st # Streamlit for UI
from pypdf import PdfReader # For reading PDF files
import os # For operating system interactions (e.g., file paths, environment variables)

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
try:
    os.environ['GOOGLE_API_KEY'] = st.secrets["GEMINI_API_KEY"] # Retrieves API key from Streamlit secrets
except KeyError:
    st.error("Gemini API key not found. Please add it to your .streamlit/secrets.toml file.", icon="🚨")
    st.stop() # Halts app execution if key is missing

Imports: Essential libraries like streamlit for the UI, pypdf for PDF parsing, and os for environment variables are imported at the top.

st.set_page_config: Configures the browser tab's title, favicon, and the overall layout of the Streamlit app.

API Key Management: This is a critical security and operational step.

It attempts to fetch the GEMINI_API_KEY from Streamlit's st.secrets. This is the recommended secure way to handle sensitive information in Streamlit apps.

The key is then set as an environment variable GOOGLE_API_KEY, which is what the langchain-google-genai library expects to find.

If the key isn't found, an error message is displayed, and st.stop() is called to prevent the app from crashing later due to a missing key.

2. Helper Functions (Lazy Loading)
This section contains the core logic for document processing and AI interaction. A key design choice here is lazy loading of AI-related libraries. This means libraries like langchain.text_splitter, langchain_google_genai, and FAISS are imported inside the functions that use them, rather than at the very top of the script.

Benefit of Lazy Loading: If there's an issue with one of these heavy AI libraries (e.g., a missing dependency, an import error), the Streamlit app can still start up and display basic UI elements (like the API key error, or even the main title) before hitting the problematic code path. This provides a more graceful failure and better user experience.

get_text_from_files(uploaded_files)
def get_text_from_files(uploaded_files):
    """Reads and extracts text from a list of uploaded files (PDF or TXT)."""
    text = ""
    for uploaded_file in uploaded_files:
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension.lower() == ".pdf":
            try:
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or "" # Extracts text from each PDF page
            except Exception as e:
                st.error(f"Error reading PDF {uploaded_file.name}: {e}", icon="📄")
        elif file_extension.lower() == ".txt":
            text += uploaded_file.read().decode("utf-8") # Reads content of TXT file
    return text

Purpose: This function is responsible for the initial data extraction from user-uploaded files.

File Handling: It iterates through uploaded_files (Streamlit's file uploader returns file-like objects).

PDF Processing: For .pdf files, it uses pypdf.PdfReader to go through each page and extract text. The or "" handles cases where a page might return None for text.

TXT Processing: For .txt files, it simply reads the decoded UTF-8 content.

Error Handling: Includes try-except blocks to catch errors during file reading and displays a Streamlit error message to the user.

get_text_chunks(text)
def get_text_chunks(text):
    """Splits text into smaller chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter # Lazy Import
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

Purpose: Breaks down the large raw text extracted from documents into smaller, manageable pieces. This is crucial because:

LLM Context Limits: Large Language Models have a limited input token size (context window).

Embedding Model Limits: Embedding models also have limits on how much text they can embed at once.

Improved Retrieval: Smaller chunks are more likely to be relevant to a specific question, leading to better retrieval accuracy.

Algorithm: RecursiveCharacterTextSplitter attempts to split text hierarchically by paragraphs, sentences, etc., falling back to characters if needed.

chunk_size=10000: Each chunk aims to be around 10,000 characters.

chunk_overlap=1000: Chunks will overlap by 1,000 characters. This helps maintain context across chunk boundaries, so if an answer spans two chunks, both might be retrieved.

get_vector_store(text_chunks)
def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings # Lazy Import
    from langchain_community.vectorstores import FAISS # Lazy Import (FAISS moved to langchain-community)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Initializes Google's embedding model
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) # Creates vector store
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}", icon="🚨") # Improved error icon
        return None

Purpose: Converts text chunks into numerical representations (embeddings) and stores them in a searchable database. This is the "Retrieval" part of RAG.

GoogleGenerativeAIEmbeddings: This class from langchain_google_genai interfaces with Google's text embedding models. models/embedding-001 is specified as the model.

FAISS (Facebook AI Similarity Search): This is a library for efficient similarity search and clustering of dense vectors.

FAISS.from_texts(text_chunks, embedding=embeddings): This method takes your text chunks, uses the embeddings model to convert each chunk into its vector representation, and then builds a FAISS index on these vectors.

Error Handling: Catches potential errors during embedding generation or vector store creation, displaying a user-friendly message.

get_conversational_chain()
def get_conversational_chain():
    """Creates the question-answering chain."""
    from langchain.prompts import PromptTemplate # Lazy Import
    from langchain_google_genai import ChatGoogleGenerativeAI # Lazy Import
    from langchain.chains.question_answering import load_qa_chain # Lazy Import

    prompt_template = """
    You are a helpful AI assistant. Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3) # Initializes Google's LLM
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) # Creates prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # Builds the QA chain
    return chain

Purpose: Configures the Large Language Model (LLM) and the prompt it will use to generate answers based on retrieved context. This is the "Generation" part of RAG.

PromptTemplate: Defines the structure of the input given to the LLM. It clearly separates the Context (retrieved document chunks) and the Question and instructs the LLM on its role.

The instruction "If the answer is not in the provided context, just say, 'The answer is not available in the context.'" is a crucial part of RAG, preventing the LLM from hallucinating answers.

ChatGoogleGenerativeAI: Initializes the Gemini 1.5 Flash model (gemini-1.5-flash-latest).

temperature=0.3: A lower temperature makes the model's output more focused and less random, suitable for factual Q&A.

load_qa_chain: This LangChain utility builds a chain for question answering.

chain_type="stuff": This means all relevant documents found during retrieval will be "stuffed" (concatenated) into the single prompt for the LLM. This is simple but effective for smaller numbers of relevant documents that fit within the LLM's context window.

3. Main Streamlit Application (main() function)
# ==============================================================================
# 5. MAIN APP FUNCTION
# ==============================================================================
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
if __name__ == "__main__":
    main()

4. Script Execution
# ==============================================================================
# 6. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    main()

if __name__ == "__main__":: This standard Python idiom ensures that the main() function is called only when the script is executed directly (e.g., streamlit run app.py), and not when it's imported as a module into another script.

🚧 Challenges Encountered and How They Were Addressed
Building this RAG agent involved several common challenges, especially when integrating with Streamlit and LangChain.

ModuleNotFoundError and Package Versioning

Challenge: Frequent changes and updates within the LangChain ecosystem often lead to modules being renamed, moved between packages (e.g., from langchain to langchain_community), or requiring specific versions of dependencies. This caused errors like No module named 'langchain_google_genai' or Module langchain_community.vectorstores not found.

Solution: Strict dependency management and keeping up-to-date with LangChain's release notes. The fix involved explicitly installing or upgrading the correct packages (e.g., pip install langchain-google-genai and pip install -U langchain-community) to ensure all necessary modules are present and correctly mapped. This also highlighted the importance of a virtual environment to isolate project dependencies.

API Key Management and Security

Challenge: Hardcoding API keys directly in the script is insecure and makes sharing difficult.

Solution: Leveraging Streamlit's st.secrets. This provides a secure and easy way to manage API keys, keeping them out of the main codebase and allowing the app to be deployed without exposing sensitive credentials. An explicit check for the API key at startup with st.stop() ensures the app doesn't proceed without proper authentication.

Streamlit's Reruns and State Management

Challenge: Streamlit reruns the entire script from top to bottom on almost every user interaction (button click, text input change). Without proper state management, variables like the vector_store would be re-created unnecessarily or lost with each rerun, leading to inefficiency or broken functionality.

Solution: Extensive use of st.session_state. By storing the vector_store and chat_history in st.session_state, their values persist across reruns, ensuring that the processed documents and ongoing conversation are maintained.

Handling Large Documents and LLM Context Limits

Challenge: Directly feeding an entire large document to an LLM for Q&A is inefficient, exceeds token limits, and often leads to poor answers as the model struggles to pinpoint relevant information.

Solution: Implementing text chunking using RecursiveCharacterTextSplitter. Breaking down documents into smaller, overlapping chunks allows the RAG system to:

Efficiently embed segments of text.

Retrieve only the most relevant segments when a question is asked.

Fit the retrieved context within the LLM's input window.

The chunk_size and chunk_overlap parameters were chosen as a balance between maintaining context and keeping chunks manageable.

Robust Error Handling in UI

Challenge: Without proper error handling, issues during PDF reading, vector store creation, or LLM calls could lead to a blank screen or a cryptic traceback for the user.

Solution: Implementing try-except blocks around potentially problematic operations (PdfReader, FAISS.from_texts, chain(...)). When an error occurs, st.error() is used to display a user-friendly message directly in the Streamlit UI, guiding the user on what went wrong and avoiding a full app crash. The use of specific emojis (📄, 🚨, 🔥, ⚠️, ✅, 📁) enhances visual feedback.
