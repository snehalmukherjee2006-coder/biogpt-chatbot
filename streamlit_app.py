import os
import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(
    page_title="BioGPT Medical Assistant",
    page_icon="🧬",
    layout="centered"
)

# 2. Sidebar with Logo and Info
with st.sidebar:
    try:
        st.image("logo.png", width=200)
    except:
        st.warning("Logo 'logo.png' not found.")

    st.title("Navigation & Info")
    st.markdown("---")
    st.header("How to use")
    st.write("1. Enter query.\n2. Review insights.\n3. *Educational use only.*")
    st.markdown("---")
    st.caption("Created by: **Snehal Mukherjee, Renisha Gracelin, Bhumiga, Lavanaya**")

# 3. Main Title
st.title("🧬 BioGPT Medical Assistant")
st.markdown("---")

# 4. Load BioGPT and Vector Database
@st.cache_resource
def load_rag_system():
    # Load BioGPT model
    chatbot = pipeline('text-generation', model='microsoft/biogpt')

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = None

    # Try loading existing FAISS index first
    if os.path.exists("faiss_medical_index"):
        try:
            vector_db = FAISS.load_local(
                "faiss_medical_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("Database loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load index: {e}")

    # If no index found, try building from medical_pdfs folder
    if vector_db is None:
        if os.path.exists("medical_pdfs") and len(os.listdir("medical_pdfs")) > 0:
            try:
                with st.spinner("Building database from PDFs... (this may take a few minutes)"):
                    loader = PyPDFDirectoryLoader("medical_pdfs")
                    documents = loader.load()

                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(documents)

                    vector_db = FAISS.from_documents(chunks, embeddings)
                    vector_db.save_local("faiss_medical_index")
                    st.success("Database built and saved successfully!")
            except Exception as e:
                st.error(f"Error building database: {e}")
        else:
            st.warning("No PDF database found. Answering from BioGPT knowledge only.")

    return chatbot, vector_db

with st.spinner("Loading BioGPT & Medical Database..."):
    chatbot, vector_db = load_rag_system()

# 5. Initialize Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 6. User Input Logic
user_input = st.chat_input('Type your health or biology question here...')

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        with st.spinner('Searching database and thinking...'):

            # RAG LOGIC: Retrieve Context from PDFs if available
            context = ""
            if vector_db:
                search_results = vector_db.similarity_search(user_input, k=2)
                context = "\n".join([doc.page_content for doc in search_results])

            # Build structured prompt
            structured_prompt = f"Context: {context}\nQuestion: {user_input}\nAnswer:"

            # Generate answer
            raw_output = chatbot(
                structured_prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.4,
                num_return_sequences=1
            )

            full_text = raw_output[0]['generated_text']

            # Clean extraction
            if "Answer:" in full_text:
                clean_answer = full_text.split("Answer:")[-1].strip()
            else:
                clean_answer = full_text.replace(structured_prompt, "").strip()

            # Remove leftover prefixes
            clean_answer = clean_answer.split("Question:")[0].strip()
            clean_answer = clean_answer.split("Context:")[0].strip()

            # Fallback
            if not clean_answer or len(clean_answer) < 5:
                clean_answer = "I'm sorry, I couldn't find a clear answer. Please try rephrasing your question."

            st.write(clean_answer)
            st.session_state.messages.append({'role': 'assistant', 'content': clean_answer})

st.markdown("---")
st.caption("⚠️ **Disclaimer:** Not a substitute for professional medical advice.")

