"""
Hapi 2: ndërtimi i bazës së njohurive (Faza A e arkitekturës).

Lexon PDF-të, i ndan në segmente, gjeneron embeddings dhe i ruan në ChromaDB.
Çdo ekzekutim e ndërton bazën nga e para. Për një korpus me dhjetëra dokumente
kjo kushton pak cent dhe shmang segmente të dyfishta. Kur shton dokumente të
reja te data/, thjesht ekzekuto përsëri këtë skript.

    python src/build_index.py
"""

import sys
import time

import chromadb

import config
import llm
from extract_and_chunk import load_and_chunk_all_pdfs


def build_index() -> int:
    print(f"Duke lexuar dokumentet nga: {config.DATA_DIR}")
    chunks = load_and_chunk_all_pdfs(config.DATA_DIR)
    if not chunks:
        print("Nuk u gjet asnjë dokument te data/. Vendosi PDF-të atje dhe provo përsëri.")
        return 0

    print(f"\nTotali: {len(chunks)} segmente. Duke gjeneruar embeddings ({config.EMBEDDING_MODEL})...")
    started = time.perf_counter()

    # Fillimisht gjenerohen të gjithë vektorët. Baza ekzistuese preket vetëm
    # nëse kjo pjesë përfundon me sukses, që një dështim (internet, çelës)
    # të mos shkatërrojë një bazë që funksiononte.
    vectors: list[list[float]] = []
    batch_size = config.EMBED_BATCH_SIZE
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(llm.embed_texts([c["text"] for c in batch]))
        print(f"  {min(start + batch_size, len(chunks))}/{len(chunks)} embeddings të gjeneruar")

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # koleksioni nuk ekzistonte ende
    collection = client.create_collection(config.COLLECTION_NAME)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[start : start + batch_size],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "program": c["program"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )

    elapsed = time.perf_counter() - started
    print(f"\nPërfundoi. Baza përmban {collection.count()} segmente ({elapsed:.1f} s).")
    print(f"Baza është ruajtur te: {config.DB_DIR}")
    return collection.count()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        build_index()
    except llm.MissingApiKeyError as error:
        print(f"\nGabim: {error}")
        sys.exit(1)
