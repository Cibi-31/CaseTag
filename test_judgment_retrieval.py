import os
import pickle
import requests
import numpy as np
import faiss

# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "vectorstore/judgments/judgments_faiss.index"
METADATA_PATH = "vectorstore/judgments/judgments_metadata.pkl"

EMBEDDING_URL = "http://127.0.0.1:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text:latest"

TOP_K = 5


# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):

    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return np.array(
        response.json()["embedding"],
        dtype=np.float32
    )


# ============================================================
# LOAD VECTORSTORE
# ============================================================

print("=" * 70)
print("CASETAG - Previous Judgment Retrieval")
print("=" * 70)

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"\nFAISS vectors : {index.ntotal}")
print(f"Metadata      : {len(metadata)}")

print("\nType your legal query.")
print("Type 'exit' to stop.")
print("=" * 70)


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

while True:

    query = input("\nEnter your query: ").strip()

    if query.lower() == "exit":
        print("\nExiting...")
        break

    if not query:
        continue

    try:

        # ----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = get_embedding(query)

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

        faiss.normalize_L2(query_embedding)

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        print("\n" + "=" * 70)
        print("TOP MATCHING PREVIOUS JUDGMENTS")
        print("=" * 70)

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            if idx < 0:
                continue

            item = metadata[idx]

            print(f"\n#{rank}")
            print("-" * 70)

            print(
                f"Similarity : {score:.4f}"
            )

            print(
                f"File       : "
                f"{item.get('judgment_file', 'N/A')}"
            )

            print(
                f"Chunk      : "
                f"{item.get('chunk_id', 'N/A')}"
            )

            print("\nJudgment text:")

            print(
                item.get("text", "")[:1200]
            )

        print("\n" + "=" * 70)

    except Exception as e:

        print(
            f"\nERROR: {type(e).__name__}: {e}"
        )