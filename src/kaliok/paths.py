import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

TEST_DOCUMENTS = (
    PROJECT_ROOT / os.getenv("KALIOK_TEST_DOCUMENTS", "test_documents")
).resolve()
