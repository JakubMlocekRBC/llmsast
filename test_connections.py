#!/usr/bin/env python3
"""test_connections.py – Test połączenia z Qdrant i modelem embeddingowym."""

import os
import sys

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Config
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "https://10.147.18.200:1234/v1")
LM_STUDIO_KEY = os.getenv("LM_STUDIO_KEY")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-nomic-embed-text-v1.5")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "diversevul_kb")


def test_qdrant():
    print("=" * 50)
    print("TEST: Połączenie z Qdrant")
    print("=" * 50)
    print(f"  URL: {QDRANT_URL}")
    print(f"  Kolekcja: {QDRANT_COLLECTION}")

    try:
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        print(f"  [OK] Połączono z Qdrant.")
        print(f"  Dostępne kolekcje: {[c.name for c in collections.collections]}")

        # Sprawdź czy docelowa kolekcja istnieje
        names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION in names:
            info = client.get_collection(QDRANT_COLLECTION)
            print(f"  [OK] Kolekcja '{QDRANT_COLLECTION}' istnieje.")
            print(f"       Punktów: {info.points_count}")
            print(f"       Rozmiar wektora: {info.config.params.vectors.size}")
        else:
            print(f"  [WARN] Kolekcja '{QDRANT_COLLECTION}' nie istnieje.")

        return True
    except Exception as e:
        print(f"  [FAIL] Nie można połączyć się z Qdrant: {e}")
        return False


def test_embedding():
    print()
    print("=" * 50)
    print("TEST: Model embeddingowy (POST /v1/embeddings)")
    print("=" * 50)

    # Budujemy URL do endpointu embeddings
    base = LM_STUDIO_URL.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/embeddings"
    else:
        url = f"{base}/v1/embeddings"

    print(f"  URL: {url}")
    print(f"  Model: {EMBEDDING_MODEL_ID}")
    print(f"  Oczekiwany rozmiar wektora: {VECTOR_SIZE}")

    api_key = LM_STUDIO_KEY
    if api_key.lower().startswith("bearer "):
        api_key = api_key.split(" ", 1)[1]

    try:
        payload = {
            "model": EMBEDDING_MODEL_ID,
            "input": "int main() { return 0; }",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        resp = httpx.post(url, json=payload, headers=headers, verify=False, timeout=120.0)
        print(f"  Status HTTP: {resp.status_code}")

        if resp.status_code != 200:
            print(f"  [FAIL] Odpowiedź: {resp.text[:500]}")
            return False

        data = resp.json()
        embedding = data["data"][0]["embedding"]

        print(f"  [OK] Embedding wygenerowany.")
        print(f"       Rozmiar wektora: {len(embedding)}")

        if len(embedding) == VECTOR_SIZE:
            print(f"  [OK] Rozmiar wektora zgodny z oczekiwanym ({VECTOR_SIZE}).")
        else:
            print(f"  [WARN] Rozmiar wektora ({len(embedding)}) != oczekiwany ({VECTOR_SIZE}).")

        return True
    except httpx.ConnectError as e:
        print(f"  [FAIL] Nie można połączyć się z serwerem: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Nie można wygenerować embeddingu: {e}")
        return False


if __name__ == "__main__":
    print("Testowanie połączeń...\n")

    qdrant_ok = test_qdrant()
    embed_ok = test_embedding()

    print()
    print("=" * 50)
    print("PODSUMOWANIE")
    print("=" * 50)
    print(f"  Qdrant:    {'OK' if qdrant_ok else 'FAIL'}")
    print(f"  Embedding: {'OK' if embed_ok else 'FAIL'}")

    if not (qdrant_ok and embed_ok):
        sys.exit(1)
    print("\nWszystkie testy przeszły pomyślnie.")
