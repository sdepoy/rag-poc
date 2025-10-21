import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp  # Import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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

Answer:
"""
prompt = PromptTemplate.from_template(template)


# --- 4. CREATE AND RUN THE RAG CHAIN ---
print("Step 4: Creating and running the RAG chain...")
# Instantiate the local LlamaCpp model
llm = LlamaCpp(
    model_path=config.MODEL_FILE_PATH, # Path to your GGUF model
    n_ctx=config.MODEL_N_CTX,          # The max sequence length to use
    temperature=config.MODEL_TEMP,     # The temperature to use for sampling
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

# --- 6. EXPLICITLY CLEAN UP ---
print("Questions has been answered, time to clean up a bit and prepare for next run...")
del llm
print("Done.")