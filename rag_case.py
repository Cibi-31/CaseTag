import os
import pickle

import faiss
import numpy as np
import ollama


# ============================================================
# CASETAG LEGAL AI
# CASE PDF → RAG → LLAMA 3.2
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_DIR = "vectorstore"

FAISS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "casetag_faiss.index"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.pkl"
)

CASE_TEXT_PATH = os.path.join(
    "data",
    "extracted_case.txt"
)


# ============================================================
# OLLAMA MODELS
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "llama3.2:latest"


# Number of legal provisions to retrieve

TOP_K = 5


# Maximum amount of case text sent to the LLM
# at one time.

MAX_CASE_TEXT = 12000


# ============================================================
# LOAD FAISS
# ============================================================

def load_faiss_index():

    print(
        "\nLoading FAISS index..."
    )

    if not os.path.exists(
        FAISS_PATH
    ):

        raise FileNotFoundError(
            f"""
FAISS index not found:

{FAISS_PATH}

Please run embedding.py first.
"""
        )

    index = faiss.read_index(
        FAISS_PATH
    )

    print(
        f"FAISS index loaded."
    )

    print(
        f"Total vectors: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print(
        "\nLoading legal metadata..."
    )

    if not os.path.exists(
        METADATA_PATH
    ):

        raise FileNotFoundError(
            f"""
Metadata file not found:

{METADATA_PATH}
"""
        )

    with open(
        METADATA_PATH,
        "rb"
    ) as file:

        metadata = pickle.load(
            file
        )

    print(
        f"Metadata records: "
        f"{len(metadata)}"
    )

    return metadata


# ============================================================
# LOAD EXTRACTED CASE
# ============================================================

def load_case_text():

    print(
        "\nLoading extracted case..."
    )

    if not os.path.exists(
        CASE_TEXT_PATH
    ):

        raise FileNotFoundError(
            f"""
Extracted case file not found:

{CASE_TEXT_PATH}

First run:

py pdf_processor.py
"""
        )

    with open(
        CASE_TEXT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        case_text = file.read()

    case_text = case_text.strip()

    if not case_text:

        raise ValueError(
            "The extracted case file is empty."
        )

    print(
        f"Case characters: "
        f"{len(case_text)}"
    )

    print(
        f"Case words: "
        f"{len(case_text.split())}"
    )

    return case_text


# ============================================================
# CREATE EMBEDDING
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

    embedding = embedding.reshape(
        1,
        -1
    )

    # Normalize because FAISS was created
    # using normalized vectors.

    faiss.normalize_L2(
        embedding
    )

    return embedding


# ============================================================
# CREATE SEARCH QUERIES FROM CASE
# ============================================================

def create_case_queries(case_text):

    """
    Creates several focused search queries from
    the extracted case.

    Instead of embedding the entire case as one
    huge query, we search using important legal
    aspects of the case.
    """

    queries = []

    # --------------------------------------------------------
    # Query 1: Complete case
    # --------------------------------------------------------

    queries.append(
        case_text[:3000]
    )

    # --------------------------------------------------------
    # Query 2: Legal issue
    # --------------------------------------------------------

    queries.append(
        "What Indian criminal law provisions "
        "are applicable to the alleged taking "
        "of property without consent?"
    )

    # --------------------------------------------------------
    # Query 3: Theft
    # --------------------------------------------------------

    queries.append(
        "Indian law relating to theft, "
        "dishonest taking of movable property "
        "without consent, punishment and "
        "applicable sections"
    )

    # --------------------------------------------------------
    # Query 4: Facts/evidence
    # --------------------------------------------------------

    queries.append(
        "Legal provisions applicable when a person "
        "is accused of taking another person's "
        "mobile phone without permission"
    )

    return queries


# ============================================================
# RETRIEVE FOR ONE QUERY
# ============================================================

def retrieve_for_query(
    query,
    index,
    metadata,
    top_k=TOP_K
):

    query_embedding = create_embedding(
        query
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position == -1:

            continue

        if index_position >= len(
            metadata
        ):

            continue

        results.append({
            "score": float(score),
            "metadata": metadata[
                index_position
            ]
        })

    return results


# ============================================================
# COMBINE RETRIEVED DOCUMENTS
# ============================================================

def retrieve_case_documents(
    case_text,
    index,
    metadata
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "             RETRIEVING LEGAL PROVISIONS"
    )

    print(
        "=" * 70
    )

    queries = create_case_queries(
        case_text
    )

    all_results = {}

    for query_number, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"\nSearch {query_number}/{len(queries)}"
        )

        print(
            f"Query: {query[:300]}..."
        )

        results = retrieve_for_query(
            query,
            index,
            metadata,
            TOP_K
        )

        for result in results:

            document = result[
                "metadata"
            ]

            # ------------------------------------------------
            # Use Act + Section as unique identifier
            # ------------------------------------------------

            key = (
                str(
                    document.get(
                        "act",
                        ""
                    )
                ),
                str(
                    document.get(
                        "section",
                        ""
                    )
                ),
                str(
                    document.get(
                        "chunk_number",
                        ""
                    )
                )
            )

            # Keep the highest score if
            # same document appears multiple times.

            if key not in all_results:

                all_results[key] = result

            elif result["score"] > all_results[
                key
            ]["score"]:

                all_results[key] = result

    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    final_results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Keep top results
    final_results = final_results[
        :TOP_K
    ]

    return final_results


# ============================================================
# DISPLAY RETRIEVED PROVISIONS
# ============================================================

def display_results(
    results
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "             RETRIEVED LEGAL PROVISIONS"
    )

    print(
        "=" * 70
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result[
            "metadata"
        ]

        score = result[
            "score"
        ]

        print(
            f"\n{rank}. "
            f"{document.get('act_name', '')}"
        )

        print(
            f"   Section: "
            f"{document.get('section', '')}"
        )

        print(
            f"   Heading: "
            f"{document.get('heading', '')}"
        )

        print(
            f"   Similarity: "
            f"{score:.4f}"
        )

        text = str(
            document.get(
                "text",
                ""
            )
        )

        if len(text) > 700:

            text = (
                text[:700]
                + "..."
            )

        print(
            f"   Legal Text:"
        )

        print(
            f"   {text}"
        )

        print(
            f"   Source:"
        )

        print(
            f"   {document.get('source_url', '')}"
        )


# ============================================================
# BUILD LEGAL CONTEXT
# ============================================================

def build_legal_context(
    results
):

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result[
            "metadata"
        ]

        score = result[
            "score"
        ]

        act_name = document.get(
            "act_name",
            ""
        )

        section = document.get(
            "section",
            ""
        )

        heading = document.get(
            "heading",
            ""
        )

        legal_text = document.get(
            "text",
            ""
        )

        source_url = document.get(
            "source_url",
            ""
        )

        context = f"""
LEGAL SOURCE {rank}

Act:
{act_name}

Section:
{section}

Heading:
{heading}

Legal Text:
{legal_text}

Source:
{source_url}

Retrieval Similarity:
{score:.4f}
"""

        context_parts.append(
            context
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# BUILD LLAMA PROMPT
# ============================================================

def build_prompt(
    case_text,
    legal_context
):

    # --------------------------------------------------------
    # Limit case text so the prompt doesn't become
    # unnecessarily large.
    # --------------------------------------------------------

    if len(case_text) > MAX_CASE_TEXT:

        case_for_prompt = (
            case_text[:MAX_CASE_TEXT]
            + "\n\n[CASE TEXT TRUNCATED FOR ANALYSIS]"
        )

    else:

        case_for_prompt = case_text

    prompt = f"""
You are CaseTag Legal AI.

You analyze Indian legal case documents using
retrieved legal provisions from the CaseTag
knowledge base.

IMPORTANT:
Use the retrieved legal sources as the primary
legal authority for your answer.

Do not invent legal sections or legal rules.

If the retrieved information is insufficient,
clearly say so.


============================================================
CASE DOCUMENT
============================================================

{case_for_prompt}


============================================================
RETRIEVED LEGAL PROVISIONS
============================================================

{legal_context}


============================================================
TASK
============================================================

Analyze the case document and determine the
most relevant legal provisions based on the
retrieved sources.


Provide the answer in the following format:


LEGAL ISSUE
-------------
Clearly state the main legal issue raised by
the case.


CASE FACTS
-------------
Summarize the important facts from the
uploaded case.


APPLICABLE ACT
-------------
State the most relevant Act.


RELEVANT SECTION
-------------
State the relevant section number and heading.


WHY THIS SECTION APPLIES
-------------
Explain how the facts of the case relate to
the retrieved legal provision.


EVIDENCE FROM CASE
-------------
Mention the important evidence or allegations
from the uploaded case that are relevant to
the legal issue.


OTHER RELEVANT PROVISIONS
-------------
Mention other retrieved provisions if they
are relevant.

If a retrieved provision is only loosely related,
say that clearly.


SOURCE
-------------
Provide the source URL of the relevant
legal provision.


LIMITATIONS
-------------
Mention if the available retrieved sources
are insufficient to reach a conclusion.


DISCLAIMER
-------------
This AI-generated analysis is for informational
and educational purposes only and is not a
substitute for advice from a qualified legal
professional.
"""

    return prompt


# ============================================================
# GENERATE LLAMA RESPONSE
# ============================================================

def generate_analysis(
    prompt
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "             GENERATING CASE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nLLM: {LLM_MODEL}"
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    # --------------------------------------------------------
    # New Ollama response object
    # --------------------------------------------------------

    if hasattr(
        response,
        "message"
    ):

        message = response.message

        if hasattr(
            message,
            "content"
        ):

            return message.content

    # --------------------------------------------------------
    # Dictionary response
    # --------------------------------------------------------

    if isinstance(
        response,
        dict
    ):

        message = response.get(
            "message",
            {}
        )

        if isinstance(
            message,
            dict
        ):

            content = message.get(
                "content",
                ""
            )

            if content:

                return content

    raise RuntimeError(
        "Could not extract Llama response."
    )


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_result(
    answer
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "                 CASETAG RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "\n"
        + answer
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# MAIN CASE ANALYSIS
# ============================================================

def analyze_case():

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    index = load_faiss_index()

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # Verify vector/metadata alignment
    # --------------------------------------------------------

    if index.ntotal != len(
        metadata
    ):

        raise RuntimeError(
            f"""
FAISS and metadata mismatch.

FAISS vectors:
{index.ntotal}

Metadata:
{len(metadata)}
"""
        )

    print(
        "\nFAISS and metadata are aligned."
    )

    # --------------------------------------------------------
    # Load case
    # --------------------------------------------------------

    case_text = load_case_text()

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results = retrieve_case_documents(
        case_text,
        index,
        metadata
    )

    if not results:

        raise RuntimeError(
            "No relevant legal provisions were retrieved."
        )

    # --------------------------------------------------------
    # Display retrieval
    # --------------------------------------------------------

    display_results(
        results
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    legal_context = build_legal_context(
        results
    )

    # --------------------------------------------------------
    # Build LLM prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        case_text,
        legal_context
    )

    # --------------------------------------------------------
    # Generate analysis
    # --------------------------------------------------------

    answer = generate_analysis(
        prompt
    )

    # --------------------------------------------------------
    # Display final answer
    # --------------------------------------------------------

    display_final_result(
        answer
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "             CASETAG LEGAL AI"
    )

    print(
        "        CASE PDF → RAG → LLAMA 3.2"
    )

    print(
        "=" * 70
    )

    try:

        analyze_case()

    except KeyboardInterrupt:

        print(
            "\n\nCase analysis interrupted."
        )

    except Exception as e:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            f"\n{e}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()