import os
import config
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. --- LOAD THE CLEANED TEXT FROM A FILE ---
file_path = config.CLEANED_DATA_FILE
print(f"Step 1: Loading data from {file_path}...")
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        final_cleaned_text = file.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please create it and add your text.")
    exit()


# 2. --- SPLIT THE TEXT INTO CHUNKS ---
print("Step 2: Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # The max number of characters in a chunk
    chunk_overlap=200, # The number of characters to overlap between chunks
    length_function=len
)
docs = text_splitter.create_documents(
    texts=[final_cleaned_text], 
    metadatas=[{"source": config.CLEANED_DATA_FILE}] 
)

print(f"   - Created {len(docs)} document chunks.")
print(f"   - Sample metadata: {docs[0].metadata}")


# 3. --- CLEAR OLD DB AND CREATE NEW ONE ---
print("Step 3: Creating embeddings and storing in ChromaDB...")
persist_directory = config.CHROMA_DIRECTORY

# Optional: Delete the old DB folder to ensure a fresh start
# This prevents "double" data if you run the script multiple times
if os.path.exists(persist_directory):
    print(f"   - Clearing old database in {persist_directory}...")
    shutil.rmtree(persist_directory)

# Use the HuggingFace embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use Chroma to create the vector store from the documents
# This single command creates the embeddings and stores them in the database
vectorstore = Chroma.from_documents(  
    documents=docs, 
    embedding=embeddings_model, 
    persist_directory=persist_directory
)

print("\n--------------------------------------------------")
print("✅ Ingestion Complete!")
print(f"Vector store created in '{persist_directory}' with {len(docs)} document chunks.")
print("--------------------------------------------------")
