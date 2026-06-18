"""Diagnose Chroma index state outside Streamlit."""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.indexing import get_chroma_client, index_exists, probe_chroma_index, _chroma_settings

print("PROJECT_ROOT:", PROJECT_ROOT)
print("chroma_persist_dir:", settings.chroma_persist_dir)
print("dir exists:", settings.chroma_persist_dir.exists())
print("sqlite exists:", (settings.chroma_persist_dir / "chroma.sqlite3").exists())

try:
    client = get_chroma_client()
    print("client ok")
    collections = [str(c) for c in client.list_collections()]
    print("collections:", collections)
    if settings.chroma_collection_name in collections:
        col = client.get_collection(settings.chroma_collection_name)
        print("collection count:", col.count())
        sample = col.peek(limit=1)
        print("peek ids:", sample.get("ids"))
    else:
        print("collection missing:", settings.chroma_collection_name)
except Exception:
    traceback.print_exc()

print("index_exists():", index_exists())
print("probe:", probe_chroma_index())
print("chroma settings:", _chroma_settings())