from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp  # Import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# --- 1. LOAD THE VECTOR STORE ---
print("Step 1: Loading vector store...")
persist_directory = 'chroma_db'
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings_model
)

# --- 2. CREATE A RETRIEVER ---
print("Step 2: Creating retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- 3. CREATE THE RAG PROMPT TEMPLATE ---
print("Step 3: Creating RAG prompt template...")
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)


# --- 4. CREATE AND RUN THE RAG CHAIN ---
print("Step 4: Creating and running the RAG chain...")
# Instantiate the local LlamaCpp model
llm = LlamaCpp(
    model_path="./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", # Path to your GGUF model
    n_ctx=2048,          # The max sequence length to use
    temperature=0.3,     # The temperature to use for sampling
    n_gpu_layers=0,      # The number of layers to offload to GPU (0 for CPU)
    verbose=False        # Suppress verbose output
)

# The RAG chain stays the same
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- 5. ASK A QUESTION ---
print("--------------------------------------------------")
question = input("Ask question: ") 
print(f"Asking question: {question}")
print("Generating answer... (this may take a moment)")

response = rag_chain.invoke(question)

print("\nAnswer:")
print(response)
print("--------------------------------------------------")