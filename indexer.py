from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print('Loading PDFs from medical_pdfs folder...')
loader = PyPDFDirectoryLoader('medical_pdfs')
documents = loader.load()
print(f'Loaded {len(documents)} pages')

if not documents:
    print('No PDFs found! Add PDFs to the medical_pdfs folder first.')
    exit()

print('Splitting text into chunks...')
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f'Created {len(chunks)} chunks')

print('Creating embeddings using sentence-transformers/all-MiniLM-L6-v2...')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

print('Building FAISS index...')
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local('faiss_medical_index')
print('Done! faiss_medical_index folder created successfully.')
