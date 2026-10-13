import os
import re
import pickle
import numpy as np
import faiss
import requests

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "judgment_data"
OUTPUT_DIR = "vectorstore/judgments"

EMBEDDING_URL = "http://127.0.0.1:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    text = clean_text(text)

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if len(chunk) >= 200:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# GET EMBEDDING FROM OLLAMA
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

    data = response.json()

    return np.array(
        data["embedding"],
        dtype=np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CASETAG - Previous Judgments FAISS Builder")
    print("=" * 70)

    if not os.path.exists(INPUT_DIR):
        print(f"\nERROR: Folder not found: {INPUT_DIR}")
        return

    txt_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".txt")
    ]

    txt_files.sort()

    print(f"\nJudgment text files found: {len(txt_files)}")

    if len(txt_files) == 0:
        print("No judgment text files found.")
        return

    all_embeddings = []
    metadata = []

    total_chunks = 0

    # --------------------------------------------------------
    # PROCESS JUDGMENTS
    # --------------------------------------------------------

    for i, filename in enumerate(txt_files, start=1):

        txt_path = os.path.join(INPUT_DIR, filename)

        print("\n" + "-" * 70)
        print(f"[{i}/{len(txt_files)}] {filename}")

        try:

            with open(
                txt_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:
                text = f.read()

            text = clean_text(text)

            if len(text) < 200:
                print("Skipped - insufficient text")
                continue

            chunks = chunk_text(text)

            print(
                f"Characters: {len(text):,}"
            )

            print(
                f"Chunks: {len(chunks)}"
            )

            # ------------------------------------------------
            # EMBEDDING EACH CHUNK
            # ------------------------------------------------

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                embedding = get_embedding(chunk)

                all_embeddings.append(embedding)

                metadata.append({
                    "judgment_file": filename,
                    "chunk_id": chunk_number,
                    "text": chunk
                })

                total_chunks += 1

                print(
                    f"\rEmbedded chunks: {total_chunks}",
                    end=""
                )

        except Exception as e:

            print(
                f"\nERROR processing {filename}: {e}"
            )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("CREATING FAISS INDEX")
    print("=" * 70)

    if not all_embeddings:

        print("ERROR: No embeddings were generated.")
        return

    embeddings = np.vstack(all_embeddings)

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # --------------------------------------------------------
    # SAVE FAISS
    # --------------------------------------------------------

    index_path = os.path.join(
        OUTPUT_DIR,
        "judgments_faiss.index"
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "judgments_metadata.pkl"
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        metadata_path,
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("JUDGMENT VECTORSTORE CREATED")
    print("=" * 70)

    print(
        f"\nJudgment files processed : {len(txt_files)}"
    )

    print(
        f"Total chunks            : {len(metadata)}"
    )

    print(
        f"Embedding dimension     : {dimension}"
    )

    print(
        f"FAISS vectors            : {index.ntotal}"
    )

    print(
        f"\nFAISS saved to:"
    )

    print(index_path)

    print(
        "\nMetadata saved to:"
    )

    print(metadata_path)

    print("\nDone! 🔥")


if __name__ == "__main__":
    main()