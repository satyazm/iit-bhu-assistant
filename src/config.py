import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT_DIR / "data" / "documents"
CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"

CATEGORIES = ["academic", "hostel", "placement", "custom"]

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
