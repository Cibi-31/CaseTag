import os
import pickle
import re
import json

import faiss
import numpy as np
import requests

from dotenv import load_dotenv

# ============================================================
# CASETAG
# PREVIOUS JUDGMENT RETRIEVAL + GROQ ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    ),
    override=True
)

print("=" * 70)
print("CASETAG - PREVIOUS JUDGMENT RETRIEVAL")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

INDEX_PATH = os.path.join(
    BASE_DIR,
    "vectorstore",
    "judgments",
    "judgments_faiss.index"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "vectorstore",
    "judgments",
    "judgments_metadata_enriched.pkl"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = (
    "http://127.0.0.1:11434/api/embeddings"
)

EMBEDDING_MODEL = (
    "nomic-embed-text:latest"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

RETRIEVAL_K = 80

FINAL_K = 5

MIN_SIMILARITY = 0.20

BAIL_PENALTY = 0.10


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS...")

if not os.path.exists(
    INDEX_PATH
):

    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}"
    )

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading enriched metadata...")

if not os.path.exists(
    METADATA_PATH
):

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_PATH}"
    )

with open(
    METADATA_PATH,
    "rb"
) as f:

    metadata = pickle.load(f)

print(
    f"Metadata records: {len(metadata)}"
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# JSON-SAFE VALUE CONVERTER
# ============================================================

def clean_json_value(value):
    """
    Convert NaN, Infinity, NumPy values and nested
    structures into JSON-safe Python values.

    FastAPI / Starlette uses strict JSON serialization,
    so Python float('nan') cannot be returned directly.
    """

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------

    if value is None:

        return None

    # --------------------------------------------------------
    # NumPy scalar
    # --------------------------------------------------------

    if isinstance(
        value,
        np.generic
    ):

        value = value.item()

    # --------------------------------------------------------
    # Float
    # --------------------------------------------------------

    if isinstance(
        value,
        float
    ):

        if not np.isfinite(
            value
        ):

            return ""

        return value

    # --------------------------------------------------------
    # Integer / Boolean
    # --------------------------------------------------------

    if isinstance(
        value,
        (int, bool)
    ):

        return value

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        value,
        dict
    ):

        return {

            str(key):
            clean_json_value(
                val
            )

            for key, val
            in value.items()
        }

    # --------------------------------------------------------
    # List / Tuple
    # --------------------------------------------------------

    if isinstance(
        value,
        (list, tuple)
    ):

        return [

            clean_json_value(
                item
            )

            for item
            in value
        ]

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(
        value,
        str
    ):

        return value

    # --------------------------------------------------------
    # Anything else
    # --------------------------------------------------------

    return str(
        value
    )


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    query
):

    query = normalize_text(
        query
    )

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )

    response = requests.post(

        OLLAMA_URL,

        json={
            "model": EMBEDDING_MODEL,
            "prompt": query
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    embedding = data.get(
        "embedding"
    )

    if embedding is None:

        raise ValueError(
            "Ollama response did not contain an embedding."
        )

    vector = np.array(
        embedding,
        dtype="float32"
    ).reshape(
        1,
        -1
    )

    faiss.normalize_L2(
        vector
    )

    return vector


# ============================================================
# SAFE FIELD GETTER
# ============================================================

def get_value(
    item,
    *keys,
    default=""
):

    if not isinstance(
        item,
        dict
    ):

        return default

    for key in keys:

        value = item.get(
            key
        )

        if value is None:

            continue

        # NumPy scalar
        if isinstance(
            value,
            np.generic
        ):

            value = value.item()

        # NaN / Infinity
        if isinstance(
            value,
            float
        ):

            if not np.isfinite(
                value
            ):

                continue

        value = normalize_text(
            value
        )

        if value:

            return value

    return default


# ============================================================
# FIND COURT ORDER / FINAL DECISION
# ============================================================

def find_order_chunks(
    chunks
):

    if not chunks:

        return []

    # --------------------------------------------------------
    # STRONG ORDER PHRASES
    # --------------------------------------------------------

    strong_order_phrases = [

        # Bail
        "bail application is allowed",
        "bail application is hereby allowed",
        "bail application stands allowed",

        "bail application is dismissed",
        "bail application is hereby dismissed",
        "bail application stands dismissed",

        "anticipatory bail is allowed",
        "anticipatory bail is hereby allowed",
        "anticipatory bail stands allowed",

        "anticipatory bail is dismissed",
        "anticipatory bail is hereby dismissed",
        "anticipatory bail stands dismissed",

        "pre-arrest bail is allowed",
        "pre-arrest bail is hereby allowed",
        "pre-arrest bail stands allowed",

        "pre-arrest bail is dismissed",
        "pre-arrest bail is hereby dismissed",
        "pre-arrest bail stands dismissed",

        "applicant be released on bail",
        "applicant shall be released on bail",
        "applicant is directed to be released on bail",
        "applicant is entitled to be released on bail",
        "let the applicant be released on bail",

        "released on bail",

        # Applications
        "application is hereby allowed",
        "application stands allowed",
        "application is allowed",

        "application is hereby dismissed",
        "application stands dismissed",
        "application is dismissed",

        # Petitions
        "petition is hereby allowed",
        "petition stands allowed",
        "petition is allowed",

        "petition is hereby dismissed",
        "petition stands dismissed",
        "petition is dismissed",

        # Appeals
        "appeal is hereby allowed",
        "appeal stands allowed",
        "appeal is allowed",

        "appeal is hereby dismissed",
        "appeal stands dismissed",
        "appeal is dismissed",

        "appeal is partly allowed",
        "appeal stands partly allowed",

        # Writ
        "writ petition is allowed",
        "writ petition is hereby allowed",
        "writ petition stands allowed",

        "writ petition is dismissed",
        "writ petition is hereby dismissed",
        "writ petition stands dismissed",

        # Criminal outcomes
        "the accused is acquitted",
        "accused is acquitted",
        "appellant is acquitted",

        "the accused is convicted",
        "accused is convicted",
        "appellant is convicted",

        "conviction is confirmed",
        "conviction is set aside",

        "sentence is confirmed",
        "sentence is set aside",

        # Disposal
        "application is disposed of",
        "petition is disposed of",
        "appeal is disposed of",
        "matter is disposed of",

        # General
        "ordered accordingly",
        "accordingly, this application",
        "accordingly, the application",
        "accordingly, the petition",
        "accordingly, the appeal"
    ]

    # --------------------------------------------------------
    # WEAK ORDER PHRASES
    # --------------------------------------------------------

    weak_order_phrases = [

        "bail is granted",
        "bail is denied",
        "bail is rejected",

        "prayer for bail is allowed",
        "prayer for bail is rejected",
        "prayer for bail is dismissed",

        "shall be released",
        "release the applicant",

        "stands dismissed",
        "stands allowed",

        "dismissed",
        "allowed",

        "disposed",
        "rejected"
    ]

    strong_results = []

    weak_results = []

    # --------------------------------------------------------
    # SEARCH CHUNKS
    # --------------------------------------------------------

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict
        ):

            continue

        text = normalize_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:

            continue

        text_lower = text.lower()

        strong_matches = [

            phrase

            for phrase
            in strong_order_phrases

            if phrase in text_lower
        ]

        weak_matches = [

            phrase

            for phrase
            in weak_order_phrases

            if phrase in text_lower
        ]

        chunk_id = chunk.get(
            "chunk_id",
            0
        )

        try:

            chunk_id = int(
                chunk_id
            )

        except Exception:

            chunk_id = 0

        if strong_matches:

            strong_results.append({

                "chunk": chunk,

                "chunk_id":
                    chunk_id,

                "matches":
                    strong_matches
            })

        elif weak_matches:

            weak_results.append({

                "chunk": chunk,

                "chunk_id":
                    chunk_id,

                "matches":
                    weak_matches
            })

    # --------------------------------------------------------
    # STRONG RESULTS
    # --------------------------------------------------------

    if strong_results:

        strong_results.sort(
            key=lambda x:
                x["chunk_id"]
        )

        return [

            item["chunk"]

            for item
            in strong_results[-3:]
        ]

    # --------------------------------------------------------
    # WEAK RESULTS
    # --------------------------------------------------------

    if weak_results:

        weak_results.sort(
            key=lambda x:
                x["chunk_id"]
        )

        return [

            item["chunk"]

            for item
            in weak_results[-3:]
        ]

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    valid_chunks = [

        chunk

        for chunk
        in chunks

        if isinstance(
            chunk,
            dict
        )
    ]

    valid_chunks.sort(

        key=lambda chunk:

        int(
            chunk.get(
                "chunk_id",
                0
            )
        )

        if str(
            chunk.get(
                "chunk_id",
                0
            )
        ).isdigit()

        else 0
    )

    return valid_chunks[-2:]


# ============================================================
# GROUP CHUNKS BY JUDGMENT
# ============================================================

def group_chunks_by_judgment():

    groups = {}

    for item in metadata:

        if not isinstance(
            item,
            dict
        ):

            continue

        judgment_file = get_value(
            item,
            "judgment_file",
            default=""
        )

        if not judgment_file:

            continue

        if judgment_file not in groups:

            groups[
                judgment_file
            ] = []

        groups[
            judgment_file
        ].append(
            item
        )

    # Sort chunks by chunk ID.
    for judgment_file in groups:

        groups[
            judgment_file
        ].sort(

            key=lambda item:

            int(
                item.get(
                    "chunk_id",
                    0
                )
            )

            if str(
                item.get(
                    "chunk_id",
                    0
                )
            ).isdigit()

            else 0
        )

    return groups


judgment_groups = (
    group_chunks_by_judgment()
)

print(
    f"Unique judgments: {len(judgment_groups)}"
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(
    text
):

    text = normalize_text(
        text
    ).lower()

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    stop_words = {

        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "a",
        "an",
        "is",
        "was",
        "were",
        "by",
        "that",
        "this",
        "it",
        "as",
        "from",
        "at",
        "be",
        "has",
        "have",
        "had",
        "been",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "which",
        "who"
    }

    return {

        token

        for token
        in tokens

        if token not in stop_words
        and len(token) > 2
    }


# ============================================================
# LEGAL CONCEPTS
# ============================================================

LEGAL_CONCEPTS = {

    "homicide": [

        "murder",
        "culpable homicide",
        "homicide",
        "death",
        "died",
        "fatal",
        "fatal injury",
        "cause of death"
    ],

    "intention": [

        "intention",
        "intent",
        "knowledge",
        "intended",
        "intentionally",
        "mens rea"
    ],

    "weapon_assault": [

        "knife",
        "weapon",
        "stabbed",
        "stabbing",
        "assault",
        "injury",
        "injuries",
        "attack",
        "blow"
    ],

    "sudden_quarrel": [

        "sudden quarrel",
        "quarrel",
        "sudden fight",
        "fight",
        "provocation",
        "grave and sudden provocation",
        "sudden altercation"
    ],

    "murder": [

        "murder",
        "murderous",
        "intention to cause death",
        "bodily injury likely to cause death"
    ],

    "culpable_homicide": [

        "culpable homicide",
        "not amounting to murder",
        "section 105",
        "section 100"
    ],

    "bail": [

        "bail",
        "anticipatory bail",
        "regular bail",
        "pre arrest bail",
        "pre-arrest bail",
        "released on bail",
        "bail application"
    ],

    "conviction": [

        "convicted",
        "conviction",
        "acquitted",
        "acquittal",
        "sentence",
        "sentenced",
        "guilty",
        "not guilty"
    ]
}


# ============================================================
# QUERY PROFILE
# ============================================================

def detect_query_profile(
    query
):

    query_lower = normalize_text(
        query
    ).lower()

    profile = {

        "is_bail_query":
            False,

        "concepts":
            set(),

        "terms":
            tokenize_text(query)
    }

    # Detect concepts.
    for concept, phrases in (
        LEGAL_CONCEPTS.items()
    ):

        for phrase in phrases:

            if phrase in query_lower:

                profile[
                    "concepts"
                ].add(
                    concept
                )

                break

    # Detect bail query.
    bail_terms = [

        "bail",
        "anticipatory bail",
        "regular bail",
        "pre arrest bail",
        "pre-arrest bail",
        "release on bail"
    ]

    for term in bail_terms:

        if term in query_lower:

            profile[
                "is_bail_query"
            ] = True

            break

    return profile


# ============================================================
# CONCEPT MATCH SCORE
# ============================================================

def calculate_concept_score(
    text,
    concepts
):

    text_lower = normalize_text(
        text
    ).lower()

    if not text_lower:

        return 0.0

    if not concepts:

        return 0.0

    matched_concepts = 0

    for concept in concepts:

        phrases = LEGAL_CONCEPTS.get(
            concept,
            []
        )

        concept_found = False

        for phrase in phrases:

            if phrase in text_lower:

                concept_found = True

                break

        if concept_found:

            matched_concepts += 1

    return (

        matched_concepts
        /
        len(concepts)
    )


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def calculate_keyword_overlap(
    query,
    text
):

    query_tokens = tokenize_text(
        query
    )

    text_tokens = tokenize_text(
        text
    )

    if not query_tokens:

        return 0.0

    if not text_tokens:

        return 0.0

    overlap = (
        query_tokens
        &
        text_tokens
    )

    return (

        len(overlap)
        /
        len(query_tokens)
    )


# ============================================================
# BAIL-DOMINANT JUDGMENT
# ============================================================

def is_bail_dominant_judgment(
    chunks
):

    if not chunks:

        return False

    combined_text = " ".join(

        normalize_text(
            chunk.get(
                "text",
                ""
            )
        )

        for chunk
        in chunks

        if isinstance(
            chunk,
            dict
        )
    ).lower()

    if not combined_text:

        return False

    bail_hits = 0

    substantive_hits = 0

    bail_phrases = [

        "bail application",
        "anticipatory bail",
        "regular bail",
        "pre-arrest bail",
        "released on bail",
        "bail bond",
        "sureties",
        "judicial custody",
        "custody"
    ]

    substantive_phrases = [

        "murder",
        "culpable homicide",
        "conviction",
        "acquitted",
        "acquittal",
        "intention to cause death",
        "knowledge that it is likely to cause death",
        "sentence",
        "convicted",
        "section 100",
        "section 103",
        "section 105"
    ]

    for phrase in bail_phrases:

        if phrase in combined_text:

            bail_hits += 1

    for phrase in substantive_phrases:

        if phrase in combined_text:

            substantive_hits += 1

    return (

        bail_hits >= 3

        and

        substantive_hits <= 2
    )


# ============================================================
# FIND BEST ISSUE EVIDENCE
# ============================================================

def find_best_issue_evidence(
    query,
    chunks,
    concepts
):

    best_chunk = None

    best_score = 0.0

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict
        ):

            continue

        text = normalize_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:

            continue

        keyword_score = (
            calculate_keyword_overlap(
                query,
                text
            )
        )

        concept_score = (
            calculate_concept_score(
                text,
                concepts
            )
        )

        combined_score = (

            0.45
            *
            keyword_score

            +

            0.55
            *
            concept_score
        )

        if (
            combined_score
            >
            best_score
        ):

            best_score = (
                combined_score
            )

            best_chunk = chunk

    return (
        best_chunk,
        best_score
    )


# ============================================================
# PREVIOUS JUDGMENT RETRIEVAL
# ============================================================

def retrieve_previous_judgments(
    query,
    k=FINAL_K
):

    query = normalize_text(
        query
    )

    if not query:

        return []

    print(
        "\nCreating query embedding..."
    )

    query_vector = (
        create_query_embedding(
            query
        )
    )

    # --------------------------------------------------------
    # QUERY PROFILE
    # --------------------------------------------------------

    query_profile = (
        detect_query_profile(
            query
        )
    )

    print(
        "\nDetected legal concepts:",
        sorted(
            query_profile[
                "concepts"
            ]
        )
    )

    print(
        "Bail query:",
        query_profile[
            "is_bail_query"
        ]
    )

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    print(
        f"\nSearching top {RETRIEVAL_K} judgment chunks..."
    )

    distances, indices = (
        index.search(
            query_vector,
            RETRIEVAL_K
        )
    )

    # --------------------------------------------------------
    # COLLECT PARENT JUDGMENTS
    # --------------------------------------------------------

    retrieved = {}

    for similarity, idx in zip(
        distances[0],
        indices[0]
    ):

        idx = int(
            idx
        )

        if idx < 0:

            continue

        if idx >= len(
            metadata
        ):

            continue

        similarity = float(
            similarity
        )

        if (
            similarity
            <
            MIN_SIMILARITY
        ):

            continue

        item = metadata[
            idx
        ]

        if not isinstance(
            item,
            dict
        ):

            continue

        judgment_file = get_value(
            item,
            "judgment_file",
            default=""
        )

        if not judgment_file:

            continue

        # Keep strongest chunk
        # for each parent judgment.

        if (
            judgment_file
            not in retrieved
        ):

            retrieved[
                judgment_file
            ] = {

                "similarity":
                    similarity,

                "matched_item":
                    item
            }

        elif (
            similarity
            >
            retrieved[
                judgment_file
            ][
                "similarity"
            ]
        ):

            retrieved[
                judgment_file
            ] = {

                "similarity":
                    similarity,

                "matched_item":
                    item
            }

    print(
        f"Candidate judgments: {len(retrieved)}"
    )

    # --------------------------------------------------------
    # ISSUE-AWARE RERANKING
    # --------------------------------------------------------

    reranked = []

    for judgment_file, item in (
        retrieved.items()
    ):

        matched_item = item[
            "matched_item"
        ]

        semantic_score = float(
            item[
                "similarity"
            ]
        )

        chunks = (
            judgment_groups.get(
                judgment_file,
                []
            )
        )

        if not chunks:

            continue

        # ----------------------------------------------------
        # BEST ISSUE EVIDENCE
        # ----------------------------------------------------

        (
            best_chunk,
            issue_score
        ) = find_best_issue_evidence(

            query,

            chunks,

            query_profile[
                "concepts"
            ]
        )

        # ----------------------------------------------------
        # BEST KEYWORD SCORE
        # ----------------------------------------------------

        best_keyword_score = 0.0

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict
            ):

                continue

            text = normalize_text(
                chunk.get(
                    "text",
                    ""
                )
            )

            score = (
                calculate_keyword_overlap(
                    query,
                    text
                )
            )

            if (
                score
                >
                best_keyword_score
            ):

                best_keyword_score = (
                    score
                )

        # ----------------------------------------------------
        # BAIL / PROCEDURAL DETECTION
        # ----------------------------------------------------

        combined_chunks_text = " ".join(

            normalize_text(
                chunk.get(
                    "text",
                    ""
                )
            )

            for chunk
            in chunks

            if isinstance(
                chunk,
                dict
            )
        ).lower()

        case_title_lower = normalize_text(
            matched_item.get(
                "case_title",
                ""
            )
        ).lower()

        docket_lower = normalize_text(
            matched_item.get(
                "docket_number",
                ""
            )
        ).lower()

        bail_phrases = [
            "bail application",
            "application for bail",
            "anticipatory bail",
            "regular bail",
            "pre-arrest bail",
            "pre arrest bail",
            "bail petition",
            "petition for bail",
            "released on bail",
            "release on bail",
            "enlarged on bail",
            "grant of bail",
            "denial of bail"
        ]

        bail_hits = sum(
            1
            for phrase
            in bail_phrases
            if phrase
            in combined_chunks_text
        )

        bail_dominant = (
            "bail" in case_title_lower
            or
            "bail" in docket_lower
            or
            bail_hits >= 2
            or
            "bail application" in combined_chunks_text
            or
            "application for bail" in combined_chunks_text
            or
            "anticipatory bail" in combined_chunks_text
            or
            is_bail_dominant_judgment(
                chunks
            )
        )

        bail_penalty = 0.0

        if (
            bail_dominant
            and
            not query_profile[
                "is_bail_query"
            ]
        ):
            # Substantive legal queries should strongly prefer
            # judgments that actually decide the legal issue.
            bail_penalty = 0.25

        elif (
            bail_hits >= 1
            and
            not query_profile[
                "is_bail_query"
            ]
        ):
            bail_penalty = 0.10

        # ----------------------------------------------------
        # SUBSTANTIVE LEGAL EVIDENCE
        # ----------------------------------------------------

        combined_chunks_text = " ".join(

            normalize_text(
                chunk.get(
                    "text",
                    ""
                )
            )

            for chunk
            in chunks

            if isinstance(
                chunk,
                dict
            )
        ).lower()

        substantive_phrases = [

            "murder",
            "culpable homicide",
            "death",
            "fatal injury",
            "intention to cause death",
            "knowledge that it is likely to cause death",
            "intention",
            "knowledge",
            "conviction",
            "acquitted",
            "acquittal",
            "convicted",
            "sentence",
            "section 100",
            "section 103",
            "section 105",
            "sudden quarrel",
            "grave and sudden provocation"
        ]

        substantive_hits = 0

        for phrase in (
            substantive_phrases
        ):

            if (
                phrase
                in combined_chunks_text
            ):

                substantive_hits += 1

        substantive_score = min(

            substantive_hits
            /
            6.0,

            1.0
        )

        # ----------------------------------------------------
        # EXPLICIT HOMICIDE-ISSUE EVIDENCE
        # ----------------------------------------------------

        homicide_issue_phrases = [
            "murder",
            "culpable homicide",
            "culpable homicide not amounting to murder",
            "not amounting to murder",
            "intention to cause death",
            "knowledge that it is likely to cause death",
            "bodily injury likely to cause death",
            "sudden quarrel",
            "sudden fight",
            "grave and sudden provocation"
        ]

        homicide_issue_hits = sum(
            1
            for phrase
            in homicide_issue_phrases
            if phrase
            in combined_chunks_text
        )

        homicide_issue_score = min(
            homicide_issue_hits
            /
            5.0,
            1.0
        )

        outcome_phrases = [
            "convicted",
            "conviction",
            "acquitted",
            "acquittal",
            "sentence",
            "sentenced",
            "guilty",
            "not guilty"
        ]

        outcome_hits = sum(
            1
            for phrase
            in outcome_phrases
            if phrase
            in combined_chunks_text
        )

        outcome_score = min(
            outcome_hits
            /
            4.0,
            1.0
        )

        substantive_bonus = 0.0

        if not query_profile["is_bail_query"]:

            if homicide_issue_hits >= 2:
                substantive_bonus += 0.10

            elif homicide_issue_hits >= 1:
                substantive_bonus += 0.05

            if outcome_hits >= 1:
                substantive_bonus += 0.08

            if (
                "sudden quarrel" in combined_chunks_text
                or
                "sudden fight" in combined_chunks_text
                or
                "grave and sudden provocation" in combined_chunks_text
            ):
                substantive_bonus += 0.08

        # ----------------------------------------------------
        # FINAL RELEVANCE SCORE
        # ----------------------------------------------------

        final_score = (

            0.20
            *
            semantic_score

            +

            0.10
            *
            best_keyword_score

            +

            0.25
            *
            issue_score

            +

            0.15
            *
            substantive_score

            +

            0.10
            *
            homicide_issue_score

            +

            0.10
            *
            outcome_score

            +

            substantive_bonus

            -

            bail_penalty
        )

        final_score = max(
            final_score,
            0.0
        )

        reranked.append({

            "judgment_file":
                judgment_file,

            "semantic_score":
                semantic_score,

            "keyword_score":
                best_keyword_score,

            "issue_score":
                issue_score,

            "substantive_score":
                substantive_score,

            "homicide_issue_score":
                homicide_issue_score,

            "outcome_score":
                outcome_score,

            "bail_hits":
                bail_hits,

            "homicide_issue_hits":
                homicide_issue_hits,

            "bail_dominant":
                bail_dominant,

            "final_score":
                final_score,

            "matched_item":
                matched_item,

            "best_chunk":
                best_chunk,

            "chunks":
                chunks
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    reranked.sort(

        key=lambda x:
            x[
                "final_score"
            ],

        reverse=True
    )

    # --------------------------------------------------------
    # SUBSTANTIVE FILTER
    # --------------------------------------------------------

    if not query_profile["is_bail_query"]:

        substantive_candidates = [

            item

            for item
            in reranked

            if (
                not item["bail_dominant"]
                and
                (
                    item["homicide_issue_score"] >= 0.20
                    or
                    item["substantive_score"] >= 0.25
                    or
                    item["outcome_score"] >= 0.25
                )
            )
        ]

        if len(substantive_candidates) >= min(3, k):

            selected_candidates = substantive_candidates[:k]

        else:

            selected_candidates = substantive_candidates[:k]

            selected_files = {
                item["judgment_file"]
                for item in selected_candidates
            }

            for item in reranked:

                if len(selected_candidates) >= k:
                    break

                if item["judgment_file"] not in selected_files:

                    selected_candidates.append(item)

                    selected_files.add(
                        item["judgment_file"]
                    )

    else:

        selected_candidates = reranked[:k]

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    final_results = []

    for item in selected_candidates:

        matched_item = item[
            "matched_item"
        ]

        judgment_file = item[
            "judgment_file"
        ]

        chunks = item[
            "chunks"
        ]

        order_chunks = (
            find_order_chunks(
                chunks
            )
        )

        best_chunk = item.get(
            "best_chunk"
        )

        if best_chunk is None:

            best_chunk = (
                matched_item
            )

        # ----------------------------------------------------
        # RESULT OBJECT
        # ----------------------------------------------------

        result = {

            "case_title":
                get_value(
                    matched_item,
                    "case_title",
                    default="Unknown"
                ),

            "court_name":
                get_value(
                    matched_item,
                    "court_name",
                    default="Unknown"
                ),

            "decision_date":
                get_value(
                    matched_item,
                    "decision_date",
                    default=""
                ),

            "docket_number":
                get_value(
                    matched_item,
                    "docket_number",
                    default=""
                ),

            "neutral_citation":
                get_value(
                    matched_item,
                    "neutral_citation",
                    default=""
                ),

            "law_report_citation":
                get_value(
                    matched_item,
                    "law_report_citation",
                    default=""
                ),

            "disposition_text":
                get_value(
                    matched_item,
                    "disposition_text",
                    default=""
                ),

            "source_pdf_s3_url":
                get_value(
                    matched_item,
                    "source_pdf_s3_url",
                    default=""
                ),

            "judgment_file":
                judgment_file,

            # ------------------------------------------------
            # SCORES
            # ------------------------------------------------

            "similarity":
                round(
                    float(
                        item[
                            "semantic_score"
                        ]
                    ),
                    4
                ),

            "relevance_score":
                round(
                    float(
                        item[
                            "final_score"
                        ]
                    ),
                    4
                ),

            "keyword_score":
                round(
                    float(
                        item[
                            "keyword_score"
                        ]
                    ),
                    4
                ),

            "issue_score":
                round(
                    float(
                        item[
                            "issue_score"
                        ]
                    ),
                    4
                ),

            "substantive_score":
                round(
                    float(
                        item[
                            "substantive_score"
                        ]
                    ),
                    4
                ),

            "homicide_issue_score":
                round(
                    float(
                        item[
                            "homicide_issue_score"
                        ]
                    ),
                    4
                ),

            "outcome_score":
                round(
                    float(
                        item[
                            "outcome_score"
                        ]
                    ),
                    4
                ),

            "bail_dominant":
                bool(
                    item[
                        "bail_dominant"
                    ]
                ),

            # ------------------------------------------------
            # MATCHED EVIDENCE
            # ------------------------------------------------

            "matched_chunk_id": (

                best_chunk.get(
                    "chunk_id"
                )

                if isinstance(
                    best_chunk,
                    dict
                )

                else

                matched_item.get(
                    "chunk_id"
                )
            ),

            "matched_text":

                normalize_text(

                    best_chunk.get(
                        "text",
                        ""
                    )

                    if isinstance(
                        best_chunk,
                        dict
                    )

                    else

                    matched_item.get(
                        "text",
                        ""
                    )
                ),

            # ------------------------------------------------
            # COURT ORDER
            # ------------------------------------------------

            "order_chunks":
                order_chunks,

            # ------------------------------------------------
            # ALL CHUNKS
            # ------------------------------------------------

            "all_chunks":
                chunks
        }

        final_results.append(
            result
        )

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ISSUE-AWARE JUDGMENT RANKING"
    )

    print(
        "=" * 70
    )

    for position, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"\n[{position}] "
            f"{result.get('case_title', 'Unknown')}"
        )

        print(
            "Court:",
            result.get(
                "court_name"
            )
        )

        print(
            "Semantic:",
            result.get(
                "similarity"
            )
        )

        print(
            "Final relevance:",
            result.get(
                "relevance_score"
            )
        )

        print(
            "Keyword:",
            result.get(
                "keyword_score"
            )
        )

        print(
            "Issue:",
            result.get(
                "issue_score"
            )
        )

        print(
            "Substantive:",
            result.get(
                "substantive_score"
            )
        )

        print(
            "Homicide issue:",
            result.get(
                "homicide_issue_score"
            )
        )

        print(
            "Outcome:",
            result.get(
                "outcome_score"
            )
        )

        print(
            "Bail dominant:",
            result.get(
                "bail_dominant"
            )
        )

    # ========================================================
    # IMPORTANT JSON SANITIZATION
    # ========================================================

    final_results = clean_json_value(
        final_results
    )

    return final_results


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(
    query,
    k=FINAL_K
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEARCHING PREVIOUS JUDGMENTS"
    )

    print(
        "=" * 70
    )

    results = (
        retrieve_previous_judgments(
            query,
            k=k
        )
    )

    print(
        f"\nRetrieved {len(results)} previous judgments."
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVED JUDGMENTS"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{i}]"
        )

        print(
            "Case:",
            result.get(
                "case_title",
                "Unknown"
            )
        )

        print(
            "Court:",
            result.get(
                "court_name",
                "Unknown"
            )
        )

        print(
            "Date:",
            result.get(
                "decision_date",
                ""
            )
        )

        print(
            "Docket:",
            result.get(
                "docket_number",
                ""
            )
        )

        print(
            "Similarity:",
            result.get(
                "similarity",
                ""
            )
        )

        print(
            "Relevance:",
            result.get(
                "relevance_score",
                ""
            )
        )

        print(
            "Matched chunk:",
            result.get(
                "matched_chunk_id",
                ""
            )
        )

        order_chunks = result.get(
            "order_chunks",
            []
        )

        print(
            "Order chunks:",

            [

                chunk.get(
                    "chunk_id"
                )

                for chunk
                in order_chunks

                if isinstance(
                    chunk,
                    dict
                )
            ]
        )

        if order_chunks:

            print(
                "\nPossible Court Order:"
            )

            for chunk in order_chunks:

                if isinstance(
                    chunk,
                    dict
                ):

                    print(

                        "\n"
                        +
                        normalize_text(
                            chunk.get(
                                "text",
                                ""
                            )
                        )
                    )

    return results


# ============================================================
# GROQ - PREVIOUS JUDGMENT ANALYSIS
# ============================================================

def generate_previous_judgment_analysis(
    case_query,
    judgments
):

    from langchain_groq import (
        ChatGroq
    )

    GROQ_MODEL = (
        "openai/gpt-oss-120b"
    )

    # --------------------------------------------------------
    # LOAD GROQ KEY ONLY WHEN CALLED
    # --------------------------------------------------------

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    if not GROQ_API_KEY:

        raise ValueError(
            "GROQ_API_KEY not found. "
            "Set GROQ_API_KEY before running CASETag."
        )

    llm = ChatGroq(

        model=GROQ_MODEL,

        temperature=0,

        api_key=GROQ_API_KEY
    )

    # ========================================================
    # HELPERS
    # ========================================================

    def clean_text(
        text
    ):

        if not text:

            return ""

        return " ".join(
            str(text).split()
        ).strip()

    def truncate_text(
        text,
        max_chars
    ):

        text = clean_text(
            text
        )

        if len(text) <= max_chars:

            return text

        return (
            text[:max_chars]
            +
            "... [truncated]"
        )

    # ========================================================
    # BUILD JUDGMENT CONTEXT
    # ========================================================

    context_parts = []

    for i, judgment in enumerate(
        judgments,
        start=1
    ):

        case_title = clean_text(
            judgment.get(
                "case_title",
                "Not available"
            )
        )

        court_name = clean_text(
            judgment.get(
                "court_name",
                "Not available"
            )
        )

        decision_date = clean_text(
            judgment.get(
                "decision_date",
                "Not available"
            )
        )

        docket_number = clean_text(
            judgment.get(
                "docket_number",
                "Not available"
            )
        )

        similarity = judgment.get(
            "similarity",
            ""
        )

        relevance_score = (
            judgment.get(
                "relevance_score",
                similarity
            )
        )

        matched_text = (
            truncate_text(

                judgment.get(
                    "matched_text",
                    ""
                ),

                2200
            )
        )

        # ----------------------------------------------------
        # COURT ORDER
        # ----------------------------------------------------

        order_chunks = (
            judgment.get(
                "order_chunks",
                []
            )
        )

        order_texts = []

        for chunk in order_chunks:

            if isinstance(
                chunk,
                dict
            ):

                text = (

                    chunk.get(
                        "text"
                    )

                    or

                    chunk.get(
                        "chunk_text"
                    )

                    or

                    chunk.get(
                        "content"
                    )

                    or

                    ""
                )

            else:

                text = str(
                    chunk
                )

            text = clean_text(
                text
            )

            if text:

                order_texts.append(
                    text
                )

        order_text = (
            "\n\n".join(
                order_texts
            )
        )

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context_parts.append(
            f"""
============================================================
PREVIOUS JUDGMENT {i}
============================================================

CASE TITLE:
{case_title}

COURT:
{court_name}

DECISION DATE:
{decision_date}

DOCKET NUMBER:
{docket_number}

FAISS SIMILARITY:
{similarity}

CASETAG RELEVANCE SCORE:
{relevance_score}

BAIL DOMINANT:
{judgment.get("bail_dominant", False)}

------------------------------------------------------------
COURT ORDER / FINAL DECISION
------------------------------------------------------------

{
    truncate_text(
        order_text,
        3200
    )
    if order_text
    else
    "NO EXPLICIT COURT ORDER FOUND"
}

------------------------------------------------------------
MOST RELEVANT RETRIEVED TEXT
------------------------------------------------------------

{matched_text}

============================================================
"""
        )

    judgment_context = (
        "\n".join(
            context_parts
        )
    )

    # ========================================================
    # GROQ PROMPT
    # ========================================================

    prompt = f"""
You are the Previous Judgment Analysis component
of CASETag Legal AI.

Your task is ONLY to extract and explain information
from the previous judgments supplied below.

You must NOT decide the current case.

CURRENT CASE:

{case_query}

PREVIOUS JUDGMENT EVIDENCE:

{judgment_context}

============================================================
IMPORTANT RULES
============================================================

RULE 1:
Use only the supplied previous-judgment evidence.

RULE 2:
The COURT ORDER / FINAL DECISION section has the highest
priority for reporting the actual outcome.

RULE 3:
The MOST RELEVANT RETRIEVED TEXT should be used to explain
the substantive legal issue when available.

RULE 4:
Do not invent facts, sections, outcomes, punishment,
reasoning or citations.

RULE 5:
Do not claim that a previous judgment determines the
result of the current case.

RULE 6:
For bail judgments, report the actual bail outcome.

Examples:

"Bail granted"

"Anticipatory bail denied"

"Bail application dismissed"

RULE 7:
Do not treat bail as punishment.

If no sentence is explicitly stated:

"Not stated"

RULE 8:
If the judgment does not actually discuss the current
legal issue, say so clearly.

RULE 9:
The "why_relevant" field must distinguish:

- factual similarity
- legal-issue similarity
- procedural similarity

RULE 10:
Do not force relevance merely because the FAISS similarity
is high.

RULE 11:
If a judgment is mainly a bail proceeding and the current
case is not a bail query, describe that limitation.

RULE 12:
Use the supplied relevance value as the relevance value.

RULE 13:
Do not invent a new numerical score.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "previous_judgments": [
        {{
            "case_title": "...",
            "court": "...",
            "decision_date": "...",
            "docket_number": "...",
            "relevance": "...",
            "relevant_issue": "...",
            "court_decision": "...",
            "outcome": "...",
            "punishment": "...",
            "why_relevant": "..."
        }}
    ]
}}
"""

    # ========================================================
    # CALL GROQ
    # ========================================================

    response = llm.invoke(
        prompt
    )

    raw_output = (
        response.content or ""
    ).strip()

    # ========================================================
    # REMOVE MARKDOWN FENCES
    # ========================================================

    if raw_output.startswith(
        "```json"
    ):

        raw_output = raw_output[
            7:
        ]

    elif raw_output.startswith(
        "```"
    ):

        raw_output = raw_output[
            3:
        ]

    if raw_output.endswith(
        "```"
    ):

        raw_output = raw_output[
            :-3
        ]

    raw_output = (
        raw_output.strip()
    )

    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        result = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        start = raw_output.find(
            "{"
        )

        end = raw_output.rfind(
            "}"
        )

        if (
            start != -1
            and end != -1
            and end > start
        ):

            json_text = raw_output[
                start:end + 1
            ]

            try:

                result = json.loads(
                    json_text
                )

            except json.JSONDecodeError:

                raise ValueError(
                    "Groq returned invalid JSON:\n"
                    +
                    raw_output
                )

        else:

            raise ValueError(
                "Groq returned invalid JSON:\n"
                +
                raw_output
            )

    # ========================================================
    # PRESERVE SOURCE / RETRIEVAL DATA
    # ========================================================

    generated = result.get(
        "previous_judgments",
        []
    )

    source_map = {}

    for judgment in judgments:

        title = normalize_text(
            judgment.get(
                "case_title",
                ""
            )
        )

        source_map[
            title
        ] = judgment

    final_previous_judgments = []

    for item in generated:

        if not isinstance(
            item,
            dict
        ):

            continue

        title = normalize_text(
            item.get(
                "case_title",
                ""
            )
        )

        original = source_map.get(
            title,
            {}
        )

        # ----------------------------------------------------
        # PRESERVE ORIGINAL SOURCE
        # ----------------------------------------------------

        item[
            "source_pdf_s3_url"
        ] = original.get(
            "source_pdf_s3_url",
            ""
        )

        item[
            "judgment_file"
        ] = original.get(
            "judgment_file",
            ""
        )

        item[
            "similarity"
        ] = original.get(
            "similarity",
            ""
        )

        item[
            "relevance_score"
        ] = original.get(
            "relevance_score",
            original.get(
                "similarity",
                ""
            )
        )

        item[
            "matched_text"
        ] = original.get(
            "matched_text",
            ""
        )

        item[
            "order_chunks"
        ] = original.get(
            "order_chunks",
            []
        )

        final_previous_judgments.append(
            item
        )

    # ========================================================
    # FINAL JSON SANITIZATION
    # ========================================================

    final_previous_judgments = (
        clean_json_value(
            final_previous_judgments
        )
    )

    return {

        "previous_judgments":
            final_previous_judgments
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_query = """
    The accused stabbed the victim with a knife during a quarrel.
    The victim sustained serious injuries and later died.
    The accused claims that there was no intention to cause death
    and that the incident occurred suddenly during the quarrel.
    The prosecution alleges that the accused intentionally attacked
    the victim and caused the fatal injuries.
    The case concerns whether the facts may attract provisions
    relating to culpable homicide, murder, or a lesser offence.
    """

    results = debug_retrieval(
        test_query,
        k=5
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST COMPLETED"
    )

    print(
        "=" * 70
    )