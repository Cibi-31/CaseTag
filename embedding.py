import os
import pickle
import time

import faiss
import numpy as np
import pandas as pd
import ollama


# ============================================================
# CASETAG LEGAL AI
# OLLAMA EMBEDDING + FAISS VECTOR DATABASE
# CHECKPOINT + RESUME VERSION
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/casetag_knowledge_base.csv"

VECTORSTORE_DIR = "vectorstore"

CHECKPOINT_DIR = os.path.join(
    VECTORSTORE_DIR,
    "checkpoint"
)

FAISS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "casetag_faiss.index"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.pkl"
)

CHECKPOINT_EMBEDDINGS = os.path.join(
    CHECKPOINT_DIR,
    "embeddings.npy"
)

CHECKPOINT_METADATA = os.path.join(
    CHECKPOINT_DIR,
    "chunks.pkl"
)


# ============================================================
# OLLAMA MODEL
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 150


# ============================================================
# CHECKPOINT SETTINGS
# ============================================================

# Save progress every 100 chunks

SAVE_EVERY = 100


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    print("=" * 60)
    print("        CASETAG OLLAMA EMBEDDING PIPELINE")
    print("=" * 60)

    print("\nLoading CaseTag knowledge base...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nKnowledge base not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df = df.fillna("")

    print(
        f"Documents loaded: {len(df)}"
    )

    required_columns = [
        "act",
        "act_name",
        "section",
        "heading",
        "text",
        "source_url"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing columns: {missing_columns}"
        )

    return df


# ============================================================
# SPLIT LONG LEGAL TEXT
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = str(text).strip()

    if not text:

        return []

    # If text is already small enough
    if len(text) <= chunk_size:

        return [text]

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end]

        # ----------------------------------------------------
        # Try to find a natural breaking point
        # ----------------------------------------------------

        if end < text_length:

            newline_position = chunk.rfind(
                "\n"
            )

            period_position = chunk.rfind(
                ". "
            )

            space_position = chunk.rfind(
                " "
            )

            best_position = max(
                newline_position,
                period_position,
                space_position
            )

            # Only use the natural break if it is
            # not too close to the beginning

            if best_position > chunk_size * 0.60:

                end = (
                    start
                    + best_position
                    + 1
                )

                chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:

            chunks.append(chunk)

        # ----------------------------------------------------
        # Move forward with overlap
        # ----------------------------------------------------

        next_start = end - overlap

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ============================================================
# PREPARE LEGAL CHUNKS
# ============================================================

def prepare_chunks(df):

    print(
        "\nPreparing legal sections..."
    )

    records = []

    for document_index, row in df.iterrows():

        act = str(
            row["act"]
        )

        act_name = str(
            row["act_name"]
        )

        section = str(
            row["section"]
        )

        heading = str(
            row["heading"]
        )

        legal_text = str(
            row["text"]
        )

        source_url = str(
            row["source_url"]
        )

        act_year = str(
            row.get("act_year", "")
        )

        jurisdiction = str(
            row.get("jurisdiction", "")
        )

        ministry = str(
            row.get("ministry", "")
        )

        department = str(
            row.get("department", "")
        )

        in_force = str(
            row.get("in_force", "")
        )

        section_count = str(
            row.get("section_count", "")
        )

        # ----------------------------------------------------
        # Split legal section into smaller chunks
        # ----------------------------------------------------

        chunks = split_text(
            legal_text
        )

        total_chunks = len(
            chunks
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            # ------------------------------------------------
            # Text sent to embedding model
            # ------------------------------------------------

            embedding_text = (
                f"Act: {act_name}\n"
                f"Act ID: {act}\n"
                f"Section: {section}\n"
                f"Heading: {heading}\n"
                f"Legal Text: {chunk}"
            )

            records.append({

                "act": act,

                "act_name": act_name,

                "section": section,

                "heading": heading,

                "text": chunk,

                "source_url": source_url,

                "act_year": act_year,

                "jurisdiction": jurisdiction,

                "ministry": ministry,

                "department": department,

                "in_force": in_force,

                "section_count": section_count,

                "chunk_number": chunk_number,

                "total_chunks": total_chunks,

                "original_document_index":
                    document_index,

                "embedding_text":
                    embedding_text
            })

    chunks_df = pd.DataFrame(
        records
    )

    print(
        f"Original sections: {len(df)}"
    )

    print(
        f"Total chunks: {len(chunks_df)}"
    )

    return chunks_df


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    print(
        "\nChecking Ollama..."
    )

    try:

        response = ollama.list()

        # ----------------------------------------------------
        # Different Ollama Python versions can return
        # slightly different objects.
        # ----------------------------------------------------

        if hasattr(
            response,
            "models"
        ):

            models = response.models

        elif isinstance(
            response,
            dict
        ):

            models = response.get(
                "models",
                []
            )

        else:

            models = []

        if not models:

            raise RuntimeError(
                "No Ollama models found."
            )

        print(
            "\nAvailable Ollama models:"
        )

        model_names = []

        for model in models:

            name = None

            if hasattr(
                model,
                "model"
            ):

                name = model.model

            elif hasattr(
                model,
                "name"
            ):

                name = model.name

            elif isinstance(
                model,
                dict
            ):

                name = (
                    model.get("model")
                    or model.get("name")
                )

            if name:

                model_names.append(
                    name
                )

                print(
                    f"  - {name}"
                )

        found = any(
            name == "nomic-embed-text"
            or name == "nomic-embed-text:latest"
            or name.startswith(
                "nomic-embed-text:"
            )
            for name in model_names
        )

        if not found:

            raise RuntimeError(
                "\n"
                "nomic-embed-text was not found.\n"
                "Run:\n"
                "ollama pull nomic-embed-text"
            )

        print(
            "\nOllama connection successful."
        )

        print(
            f"Embedding model: "
            f"{EMBEDDING_MODEL}"
        )

    except Exception as e:

        print(
            "\nOllama check failed."
        )

        print(
            f"Error: {e}"
        )

        raise


# ============================================================
# CREATE ONE EMBEDDING
# ============================================================

def create_embedding(text):

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    if "embedding" not in response:

        raise RuntimeError(
            "Ollama did not return an embedding."
        )

    embedding = np.array(
        response["embedding"],
        dtype=np.float32
    )

    return embedding


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint():

    # --------------------------------------------------------
    # No checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        CHECKPOINT_EMBEDDINGS
    ):

        return [], 0

    if not os.path.exists(
        CHECKPOINT_METADATA
    ):

        return [], 0

    print(
        "\nCheckpoint found!"
    )

    try:

        # ----------------------------------------------------
        # Load saved embeddings
        # ----------------------------------------------------

        embeddings = np.load(
            CHECKPOINT_EMBEDDINGS
        )

        # ----------------------------------------------------
        # Load saved metadata
        # ----------------------------------------------------

        with open(
            CHECKPOINT_METADATA,
            "rb"
        ) as file:

            metadata = pickle.load(
                file
            )

        completed = len(
            embeddings
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(metadata) != completed:

            raise RuntimeError(
                "Checkpoint embeddings and metadata "
                "have different lengths."
            )

        print(
            f"Previously completed: "
            f"{completed} chunks"
        )

        return (
            embeddings.tolist(),
            completed
        )

    except Exception as e:

        print(
            "\nCould not load checkpoint."
        )

        print(
            f"Error: {e}"
        )

        print(
            "Starting from the beginning."
        )

        return [], 0


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    embeddings,
    chunks_df,
    completed
):

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    embedding_array = np.array(
        embeddings,
        dtype=np.float32
    )

    np.save(
        CHECKPOINT_EMBEDDINGS,
        embedding_array
    )

    # --------------------------------------------------------
    # Save matching metadata
    # --------------------------------------------------------

    completed_metadata = (
        chunks_df
        .iloc[:completed]
        .drop(
            columns=[
                "embedding_text"
            ],
            errors="ignore"
        )
    )

    metadata = completed_metadata.to_dict(
        orient="records"
    )

    with open(
        CHECKPOINT_METADATA,
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file
        )

    print(
        f"\nCheckpoint saved: "
        f"{completed} chunks"
    )


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks_df
):

    total = len(
        chunks_df
    )

    print(
        "\nCreating embeddings using Ollama..."
    )

    print(
        f"Total chunks: {total}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    embeddings, start_index = (
        load_checkpoint()
    )

    # --------------------------------------------------------
    # Already finished
    # --------------------------------------------------------

    if start_index >= total:

        print(
            "\nAll chunks are already embedded."
        )

        return np.array(
            embeddings,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Resume or start
    # --------------------------------------------------------

    if start_index > 0:

        print(
            "\n"
            "=========================================="
        )

        print(
            f"RESUMING FROM CHUNK "
            f"{start_index + 1}"
        )

        print(
            "=========================================="
        )

    else:

        print(
            "\nStarting embedding from chunk 1"
        )

    start_time = time.time()

    # ========================================================
    # EMBEDDING LOOP
    # ========================================================

    for i in range(
        start_index,
        total
    ):

        row = chunks_df.iloc[
            i
        ]

        text = row[
            "embedding_text"
        ]

        try:

            embedding = create_embedding(
                text
            )

            embeddings.append(
                embedding
            )

        except Exception as e:

            print(
                "\n"
                "=" * 60
            )

            print(
                "EMBEDDING ERROR"
            )

            print(
                "=" * 60
            )

            print(
                f"Chunk: {i + 1}"
            )

            print(
                f"Act: {row['act_name']}"
            )

            print(
                f"Section: {row['section']}"
            )

            print(
                f"Chunk: "
                f"{row['chunk_number']}/"
                f"{row['total_chunks']}"
            )

            print(
                f"Error: {e}"
            )

            print(
                "\nTrying smaller fallback chunks..."
            )

            # ------------------------------------------------
            # FALLBACK
            # ------------------------------------------------

            smaller_chunks = split_text(
                text,
                chunk_size=600,
                overlap=50
            )

            smaller_embeddings = []

            for small_chunk in smaller_chunks:

                response = ollama.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=small_chunk
                )

                if "embedding" not in response:

                    raise RuntimeError(
                        "Ollama failed during fallback."
                    )

                small_embedding = np.array(
                    response["embedding"],
                    dtype=np.float32
                )

                smaller_embeddings.append(
                    small_embedding
                )

            if not smaller_embeddings:

                raise RuntimeError(
                    "Fallback embedding failed."
                )

            # ------------------------------------------------
            # Average fallback embeddings
            # ------------------------------------------------

            embedding = np.mean(
                smaller_embeddings,
                axis=0
            ).astype(
                np.float32
            )

            embeddings.append(
                embedding
            )

            print(
                "Fallback embedding successful."
            )

        # ----------------------------------------------------
        # Completed count
        # ----------------------------------------------------

        completed = i + 1

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if (
            completed % SAVE_EVERY == 0
            or completed == total
        ):

            elapsed = (
                time.time()
                - start_time
            )

            percentage = (
                completed
                / total
                * 100
            )

            # Avoid division by zero
            if completed > start_index:

                processed_now = (
                    completed
                    - start_index
                )

                seconds_per_chunk = (
                    elapsed
                    / processed_now
                )

                remaining = (
                    total
                    - completed
                )

                estimated_remaining_seconds = (
                    remaining
                    * seconds_per_chunk
                )

                estimated_hours = (
                    estimated_remaining_seconds
                    / 3600
                )

            else:

                estimated_hours = 0

            print(
                "\n"
                f"Progress: "
                f"{completed}/{total} "
                f"({percentage:.2f}%)"
            )

            print(
                f"Elapsed: "
                f"{elapsed / 60:.1f} minutes"
            )

            if estimated_hours > 0:

                print(
                    f"Estimated remaining: "
                    f"{estimated_hours:.2f} hours"
                )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if (
            completed % SAVE_EVERY == 0
            or completed == total
        ):

            save_checkpoint(
                embeddings,
                chunks_df,
                completed
            )

    return np.array(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(
    embeddings
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Creating FAISS index..."
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Check shape
    # --------------------------------------------------------

    if len(embeddings) == 0:

        raise RuntimeError(
            "No embeddings available."
        )

    if embeddings.ndim != 2:

        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    dimension = embeddings.shape[
        1
    ]

    print(
        f"Number of vectors: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # --------------------------------------------------------
    # Normalize embeddings
    #
    # FAISS Inner Product + normalized vectors
    # = cosine similarity
    # --------------------------------------------------------

    faiss.normalize_L2(
        embeddings
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"Vectors stored in FAISS: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# SAVE FINAL VECTORSTORE
# ============================================================

def save_vectorstore(
    index,
    chunks_df
):

    print(
        "\nSaving final vectorstore..."
    )

    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        FAISS_PATH
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_df = (
        chunks_df
        .drop(
            columns=[
                "embedding_text"
            ],
            errors="ignore"
        )
    )

    metadata = metadata_df.to_dict(
        orient="records"
    )

    with open(
        METADATA_PATH,
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file
        )

    print(
        "\n"
        "=========================================="
    )

    print(
        "FINAL VECTORSTORE SAVED"
    )

    print(
        "=========================================="
    )

    print(
        f"\nFAISS index:"
        f"\n{FAISS_PATH}"
    )

    print(
        f"\nMetadata:"
        f"\n{METADATA_PATH}"
    )


# ============================================================
# RETRIEVAL TEST
# ============================================================

def test_retrieval(
    index,
    chunks_df
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "                 RETRIEVAL TEST"
    )

    print(
        "=" * 60
    )

    query = input(
        "\nEnter a legal query "
        "(press Enter to skip): "
    ).strip()

    if not query:

        print(
            "\nRetrieval test skipped."
        )

        return

    print(
        f"\nSearching for:"
    )

    print(
        query
    )

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = create_embedding(
        query
    )

    query_embedding = query_embedding.reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Normalize query
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )

    # --------------------------------------------------------
    # Search top 5
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        5
    )

    print(
        "\nTop 5 Retrieved Results:"
    )

    print(
        "-" * 60
    )

    for rank, (
        score,
        index_position
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if index_position == -1:

            continue

        row = chunks_df.iloc[
            index_position
        ]

        print(
            f"\n{rank}. "
            f"{row['act_name']}"
        )

        print(
            f"   Section: "
            f"{row['section']}"
        )

        print(
            f"   Heading: "
            f"{row['heading']}"
        )

        print(
            f"   Chunk: "
            f"{row['chunk_number']}/"
            f"{row['total_chunks']}"
        )

        print(
            f"   Similarity: "
            f"{score:.4f}"
        )

        legal_text = str(
            row["text"]
        )

        if len(legal_text) > 500:

            legal_text = (
                legal_text[:500]
                + "..."
            )

        print(
            f"   Legal Text:"
        )

        print(
            f"   {legal_text}"
        )

        print(
            f"   Source:"
        )

        print(
            f"   {row['source_url']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # STEP 1
        # Load legal knowledge base
        # ----------------------------------------------------

        df = load_knowledge_base()

        # ----------------------------------------------------
        # STEP 2
        # Check Ollama
        # ----------------------------------------------------

        check_ollama()

        # ----------------------------------------------------
        # STEP 3
        # Prepare chunks
        # ----------------------------------------------------

        chunks_df = prepare_chunks(
            df
        )

        # ----------------------------------------------------
        # STEP 4
        # Create embeddings
        # ----------------------------------------------------

        embeddings = create_embeddings(
            chunks_df
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(embeddings) != len(
            chunks_df
        ):

            raise RuntimeError(
                "\n"
                "Embedding count does not match "
                "chunk count.\n"
                f"Embeddings: {len(embeddings)}\n"
                f"Chunks: {len(chunks_df)}"
            )

        # ----------------------------------------------------
        # STEP 5
        # Create FAISS
        # ----------------------------------------------------

        index = create_faiss_index(
            embeddings
        )

        # ----------------------------------------------------
        # STEP 6
        # Save vectorstore
        # ----------------------------------------------------

        save_vectorstore(
            index,
            chunks_df
        )

        # ----------------------------------------------------
        # STEP 7
        # Retrieval test
        # ----------------------------------------------------

        test_retrieval(
            index,
            chunks_df
        )

        # ----------------------------------------------------
        # DONE
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "       CASETAG EMBEDDING COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            "\nYour RAG vectorstore is ready."
        )

    except KeyboardInterrupt:

        print(
            "\n\n"
            + "=" * 60
        )

        print(
            "EMBEDDING INTERRUPTED"
        )

        print(
            "=" * 60
        )

        print(
            "\nYour latest checkpoint has been preserved."
        )

        print(
            "Run the same command again to resume:"
        )

        print(
            "\n    py embedding.py"
        )

    except Exception as e:

        print(
            "\n\n"
            + "=" * 60
        )

        print(
            "PROGRAM ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"\n{e}"
        )

        raise


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()