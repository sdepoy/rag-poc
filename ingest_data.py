import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. --- LOAD THE CLEANED TEXT FROM A FILE ---
file_path = "cleaned_data.txt"
print(f"Step 1: Loading data from {file_path}...")
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        final_cleaned_text = file.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please create it and add your text.")
    exit()


# 2. --- SPLIT THE TEXT INTO CHUNKS ---
print("Step 1: Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # The max number of characters in a chunk
    chunk_overlap=200, # The number of characters to overlap between chunks
    length_function=len
)
docs = text_splitter.split_text(final_cleaned_text)


# 3. --- CREATE EMBEDDINGS AND STORE IN CHROMA ---
print("Step 2: Creating embeddings and storing in ChromaDB...")
persist_directory = 'chroma_db'

# Use the HuggingFace embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use Chroma to create the vector store from the documents
# This single command creates the embeddings and stores them in the database
vectorstore = Chroma.from_texts(
    texts=docs, 
    embedding=embeddings_model, 
    persist_directory=persist_directory
)

print("\n--------------------------------------------------")
print("✅ Ingestion Complete!")
print(f"Vector store created in '{persist_directory}' with {vectorstore._collection.count()} document chunks.")
print("--------------------------------------------------")
