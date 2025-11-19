import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 1. LOAD THE VECTOR STORE ---
print("Step 1: Loading vector store...")
persist_directory = config.CHROMA_DIRECTORY
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
Use the following context to answer the question.
Answer based *only* on the context provided.
Do not add any section numbers, headers, or titles (like '4.3.2.').
Your response should be a simple paragraph or bulleted list.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)


# --- 4. LOAD LLM AND CREATE AND RUN THE RAG CHAIN ---
print("Step 4: Creating and running the RAG chain...")
# Instantiate the local LlamaCpp model
llm = LlamaCpp(
    model_path=config.MODEL_FILE_PATH, # Path to the GGUF model
    n_ctx=config.MODEL_N_CTX,          # The max sequence length to use
    temperature=config.MODEL_TEMP,     # The temperature to use for sampling
    n_gpu_layers=0,      # The number of layers to offload to GPU (0 for CPU)
    verbose=False        # Suppress verbose output
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- 5. INTERACTIVE LOOP ---
print("\n==================================================")
print("🤖 ASK ME ANYTHING")
print("Type 'exit' or 'quit' to stop the program.")
print("==================================================\n")

while True:
    # Get user input
    question = input("\n👉 Ask a question: ")
    
    # Check for exit commands
    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye! 👋")
        break
    
    if not question.strip():
        continue

    # --- A. SOURCE CITATION ---
    # We verify what the retriever sees BEFORE asking the LLM
    # This adds transparency/trust to the tool
    docs = retriever.invoke(question)
    print(f"\n[Searching {len(docs)} document chunks...]")
    
    # Print the "Source" for each found doc (assuming you have filenames in metadata)
    # If you ingested from a single txt file, this might just show IDs, 
    # but with PDFs via LlamaIndex/LangChain loaders, it shows filenames.
    print("Sources Found:")
    for i, doc in enumerate(docs):
        # Try to get the source, fallback to 'Chunk' if metadata is empty
        source = doc.metadata.get('source', f'Chunk {i+1}')
        print(f" - {source}")

    print("\nThinking...\n")
    
    # --- B. GENERATE ANSWER (STREAMING) ---
    try:
        for chunk in rag_chain.stream(question):
            # Filter out non-string chunks (like metadata) just in case
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
        print("\n") # Newline after answer is done
        
    except Exception as e:
        print(f"\n❌ Error generating response: {e}")

# --- 6. EXPLICITLY CLEAN UP ---
print("Questions have been answered, time to clean up a bit and prepare for next run...")
del llm
print("Done.")