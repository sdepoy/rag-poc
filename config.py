import os

# --- Data and Model Paths (override via environment variables) ---
CLEANED_DATA_FILE = os.environ.get("CLEANED_DATA_FILE", "./data/cleaned_data.txt")
MODEL_FILE_PATH = os.environ.get(
    "MODEL_FILE_PATH", "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
)
CHROMA_DIRECTORY = os.environ.get("CHROMA_DIRECTORY", "./chroma_db")

# --- Model Parameters ---
MODEL_N_CTX = int(os.environ.get("MODEL_N_CTX", 2048))
MODEL_TEMP = float(os.environ.get("MODEL_TEMP", 0.3))
