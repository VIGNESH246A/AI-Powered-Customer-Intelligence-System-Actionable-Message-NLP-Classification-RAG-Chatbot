import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root
BASE_DIR = Path(__file__).parent.parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data" / "rag_data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables")
    print("Please set your API key in the .env file")

# Model Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))

# Vector Store Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(VECTOR_STORE_DIR / "faiss_index"))

# LLM Configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Application Configuration
APP_NAME = os.getenv("APP_NAME", "Integrated Customer Support System")

# File Paths
KB_FILE = RAW_DATA_DIR / "kb.txt"
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"