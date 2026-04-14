from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Folder containing your PDFs
DATA_PATH = "data"

documents = []

# Load all PDFs
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
vector_db = FAISS.from_documents(documents, embeddings)

# Save locally
vector_db.save_local("faiss_medical_index")

print("✅ FAISS index created successfully!")