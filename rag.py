# ============================================================
# CASETag LEGAL AI
# RAG + FAISS + OLLAMA EMBEDDINGS + GROQ LLM
# ============================================================

import os
import re
import json
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Load .env from CASETag root folder
load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore"
)

FAISS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "casetag_faiss.index"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.pkl"
)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text:latest"

LLM_MODEL = "openai/gpt-oss-120b"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not configured")


# ------------------------------------------------------------
# Retrieval settings
# ------------------------------------------------------------

RETRIEVAL_K = 40

FINAL_K = 5

MIN_RELEVANCE = 0.20

# Minimum combined relevance required for a result to be
# considered genuinely useful.
MIN_FINAL_SCORE = 0.18

# Generic queries should not trigger random legal retrieval.
MIN_QUERY_TOKENS = 1


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    value: Any
) -> str:

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def load_vectorstore():

    if not os.path.exists(
        FAISS_PATH
    ):
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}"
        )

    if not os.path.exists(
        METADATA_PATH
    ):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    loaded_index = faiss.read_index(
        FAISS_PATH
    )

    with open(
        METADATA_PATH,
        "rb"
    ) as file:

        loaded_metadata = pickle.load(
            file
        )

    print(
        "FAISS vectors :",
        loaded_index.ntotal
    )

    print(
        "FAISS metric  :",
        loaded_index.metric_type
    )

    print(
        "Metadata      :",
        len(loaded_metadata)
    )

    if (
        loaded_index.ntotal
        != len(loaded_metadata)
    ):
        raise ValueError(
            "FAISS and metadata are NOT aligned."
        )

    print(
        "FAISS and metadata are aligned."
    )

    return (
        loaded_index,
        loaded_metadata
    )


# ============================================================
# GLOBAL VECTORSTORE
# ============================================================

index, metadata = load_vectorstore()

# main.py health endpoint may use this name
faiss_index = index


# ============================================================
# MODELS
# ============================================================

embeddings_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)



llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    max_tokens=1200,
    api_key=GROQ_API_KEY
)

print(
    f"Embedding model: {EMBEDDING_MODEL}"
)

print(
    f"LLM: Groq / {LLM_MODEL}"
)

print(
    "CASETAG RAG system loaded successfully."
)


# ============================================================
# QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    query: str
) -> np.ndarray:

    query = clean_text(
        query
    )

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    vector = embeddings_model.embed_query(
        query
    )

    vector = np.asarray(
        vector,
        dtype="float32"
    ).reshape(
        1,
        -1
    )

    # FAISS uses inner product.
    # Since both document and query vectors are normalized,
    # inner product behaves as cosine similarity.
    faiss.normalize_L2(
        vector
    )

    return vector


# ============================================================
# METADATA HELPERS
# ============================================================

def get_metadata_value(
    item: Any,
    *keys: str,
    default: str = ""
) -> str:

    if isinstance(
        item,
        dict
    ):

        for key in keys:

            if key in item:

                value = item.get(
                    key
                )

                if value is not None:

                    return clean_text(
                        value
                    )

    if hasattr(
        item,
        "metadata"
    ):

        item_metadata = getattr(
            item,
            "metadata",
            {}
        )

        if isinstance(
            item_metadata,
            dict
        ):

            for key in keys:

                if key in item_metadata:

                    value = item_metadata.get(
                        key
                    )

                    if value is not None:

                        return clean_text(
                            value
                        )

    return default


def get_text_value(
    item: Any
) -> str:

    if isinstance(
        item,
        dict
    ):

        for key in [
            "text",
            "chunk",
            "page_content",
            "content",
            "legal_text"
        ]:

            value = item.get(
                key
            )

            if value:

                return clean_text(
                    value
                )

    if hasattr(
        item,
        "page_content"
    ):

        value = getattr(
            item,
            "page_content",
            ""
        )

        if value:

            return clean_text(
                value
            )

    if hasattr(
        item,
        "metadata"
    ):

        item_metadata = getattr(
            item,
            "metadata",
            {}
        )

        if isinstance(
            item_metadata,
            dict
        ):

            for key in [
                "text",
                "chunk",
                "page_content",
                "content",
                "legal_text"
            ]:

                value = item_metadata.get(
                    key
                )

                if value:

                    return clean_text(
                        value
                    )

    return ""


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "into",
    "may",
    "might",
    "could",
    "can",
    "which",
    "what",
    "whether",
    "who",
    "whom",
    "where",
    "when",
    "how",
    "why",
    "does",
    "do",
    "did",
    "under",
    "applicable",
    "relevant",
    "section",
    "act",
    "law",
    "legal",
    "provision",
    "please",
    "tell",
    "me",
}


# ============================================================
# TOKENIZATION
# ============================================================

def useful_tokens(
    text: str
) -> List[str]:

    text = clean_text(
        text
    ).lower()

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    return [
        token
        for token in tokens
        if (
            token not in STOP_WORDS
            and len(token) >= 3
        )
    ]


# ============================================================
# QUERY TYPE DETECTION
# ============================================================

def is_punishment_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    terms = [
        "punishment",
        "punished",
        "punishable",
        "penalty",
        "sentence",
        "sentencing",
        "imprisonment",
        "jail",
        "fine",
        "death penalty",
        "life imprisonment",
        "how many years",
    ]

    return any(
        term in query
        for term in terms
    )


def is_sexual_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    terms = [
        "rape",
        "sexual assault",
        "sexual offence",
        "sexual offense",
        "sexually",
        "intercourse",
        "penetration",
        "modesty",
    ]

    return any(
        term in query
        for term in terms
    )


def is_homicide_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    terms = [
        "murder",
        "culpable homicide",
        "causing death",
        "death",
        "died",
        "killed",
        "fatal",
        "sharp object",
        "sharp weapon",
        "stab",
        "stabbed",
        "stabbing",
        "post-mortem",
        "postmortem",
    ]

    return any(
        term in query
        for term in terms
    )


def is_negligence_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    terms = [
        "negligence",
        "negligent",
        "rash driving",
        "rash or negligent",
        "road accident",
        "vehicle accident",
        "driving accident",
    ]

    return any(
        term in query
        for term in terms
    )


# ============================================================
# GENERIC / LOW INFORMATION QUERY
# ============================================================

def is_low_information_query(
    query: str
) -> bool:

    query = clean_text(
        query
    )

    if not query:
        return True

    tokens = useful_tokens(
        query
    )

    if len(tokens) < MIN_QUERY_TOKENS:
        return True

    generic_queries = {
        "what would you like to know",
        "what would you like to know?",
        "ask a question",
        "legal query",
        "legal question",
        "hello",
        "hi",
        "hey",
        "help",
        "help me",
        "tell me something",
    }

    if query.lower() in generic_queries:
        return True

    return False


# ============================================================
# EXPLICIT SECTION DETECTION
# ============================================================

def extract_requested_section(
    query: str
) -> str:

    query = clean_text(
        query
    ).lower()

    patterns = [
        r"\bsection\s+(\d+[a-z]?)\b",
        r"\bsec\.?\s*(\d+[a-z]?)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query
        )

        if match:

            return match.group(
                1
            ).lower()

    return ""


# ============================================================
# ACT DETECTION
# ============================================================

def is_explicit_ipc_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    return (
        "ipc" in query
        or "indian penal code" in query
    )


def is_explicit_bns_query(
    query: str
) -> bool:

    query = clean_text(
        query
    ).lower()

    return (
        "bns" in query
        or "bharatiya nyaya sanhita" in query
    )


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(
    query: str,
    document_text: str
) -> float:

    query_tokens = set(
        useful_tokens(
            query
        )
    )

    document_tokens = set(
        useful_tokens(
            document_text
        )
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        .intersection(
            document_tokens
        )
    )

    return (
        len(overlap)
        /
        len(query_tokens)
    )


# ============================================================
# PHRASE SCORE
# ============================================================

def phrase_score(
    query: str,
    document_text: str
) -> float:

    query = clean_text(
        query
    ).lower()

    document_text = clean_text(
        document_text
    ).lower()

    phrases = [
        "punishment for murder",
        "punishment for culpable homicide",
        "culpable homicide",
        "causing death",
        "death by negligence",
        "rash or negligent act",
        "attempt to murder",
        "grievous hurt",
        "voluntarily causing hurt",
        "sharp object",
        "sharp weapon",
        "injury resulting in death",
        "gang rape",
        "sexual assault",
        "theft",
        "robbery",
        "cheating",
        "criminal breach of trust",
        "kidnapping",
    ]

    score = 0.0

    for phrase in phrases:

        if (
            phrase in query
            and phrase in document_text
        ):

            score += 1.0

    return min(
        score / 3.0,
        1.0
    )


# ============================================================
# TOPIC BOOST
# ============================================================

def legal_topic_boost(
    query: str,
    heading: str,
    document_text: str
) -> float:

    query_tokens = set(
        useful_tokens(
            query
        )
    )

    if not query_tokens:
        return 0.0

    heading_tokens = set(
        useful_tokens(
            heading
        )
    )

    document_tokens = set(
        useful_tokens(
            document_text
        )
    )

    score = 0.0

    heading_overlap = (
        query_tokens
        .intersection(
            heading_tokens
        )
    )

    document_overlap = (
        query_tokens
        .intersection(
            document_tokens
        )
    )

    if heading_overlap:

        score += min(
            len(heading_overlap)
            * 0.10,
            0.35
        )

    if document_overlap:

        score += min(
            len(document_overlap)
            * 0.015,
            0.15
        )

    return min(
        score,
        0.50
    )


# ============================================================
# OFFENCE MATCH
# ============================================================

def offence_match_score(
    query: str,
    heading: str,
    document_text: str
) -> float:

    query = clean_text(
        query
    ).lower()

    heading = clean_text(
        heading
    ).lower()

    document_text = clean_text(
        document_text
    ).lower()

    offence_groups = [

        [
            "murder",
            "murder",
            "punishment for murder",
        ],

        [
            "culpable homicide",
            "culpable homicide",
            "punishment for culpable homicide",
        ],

        [
            "theft",
            "theft",
            "punishment for theft",
        ],

        [
            "robbery",
            "robbery",
        ],

        [
            "rape",
            "rape",
            "punishment for rape",
        ],

        [
            "sexual assault",
            "sexual assault",
        ],

        [
            "cheating",
            "cheating",
        ],

        [
            "kidnapping",
            "kidnapping",
        ],

        [
            "grievous hurt",
            "grievous hurt",
        ],
    ]

    score = 0.0

    for group in offence_groups:

        if any(
            term in query
            for term in group
        ):

            if any(
                term in heading
                for term in group
            ):

                score += 0.50

            elif any(
                term in document_text
                for term in group
            ):

                score += 0.15

    return min(
        score,
        0.60
    )


# ============================================================
# UNRELATED PENALTY
# ============================================================

def unrelated_penalty(
    query: str,
    document_text: str
) -> float:

    query = clean_text(
        query
    ).lower()

    document_text = clean_text(
        document_text
    ).lower()

    penalty = 0.0

    sexual = is_sexual_query(
        query
    )

    homicide = is_homicide_query(
        query
    )

    negligence = is_negligence_query(
        query
    )

    document_sexual = any(
        term in document_text
        for term in [
            "rape",
            "sexual assault",
            "sexual offence",
            "sexual offense",
            "intercourse",
            "penetration",
        ]
    )

    document_negligence = any(
        term in document_text
        for term in [
            "negligence",
            "negligent",
            "rash driving",
            "rash or negligent",
        ]
    )

    # Homicide query but sexual provision
    if (
        homicide
        and not sexual
        and document_sexual
    ):

        penalty += 1.0

    # Non-negligence homicide but negligence provision
    if (
        homicide
        and not negligence
        and document_negligence
    ):

        penalty += 0.35

    return min(
        penalty,
        1.0
    )
# ============================================================
# PUNISHMENT RELEVANCE
# ============================================================

def punishment_relevance_score(
    query: str,
    section: str,
    heading: str,
    document_text: str
) -> float:

    if not is_punishment_query(query):
        return 0.0

    query_lower = clean_text(query).lower()
    heading_lower = clean_text(heading).lower()

    score = 0.0

    # A heading explicitly saying punishment is highly relevant
    if any(
        word in heading_lower
        for word in [
            "punishment",
            "punishable",
            "penalty",
            "sentence",
        ]
    ):
        score += 0.45

    # Direct offence match
    if (
        "murder" in query_lower
        and "murder" in heading_lower
    ):
        score += 0.55

    if (
        "rape" in query_lower
        and "rape" in heading_lower
    ):
        score += 0.55

    if (
        "theft" in query_lower
        and "theft" in heading_lower
    ):
        score += 0.55

    if (
        "robbery" in query_lower
        and "robbery" in heading_lower
    ):
        score += 0.55

    if (
        "culpable homicide" in query_lower
        and "culpable homicide" in heading_lower
    ):
        score += 0.55

    requested_section = extract_requested_section(query)

    if (
        requested_section
        and clean_text(section).lower() == requested_section
    ):
        score += 1.0

    return min(score, 1.0)


# ============================================================
# PUNISHMENT MISMATCH
# ============================================================

def punishment_mismatch_penalty(
    query: str,
    section: str,
    heading: str,
    document_text: str
) -> float:

    if not is_punishment_query(query):
        return 0.0

    heading_lower = clean_text(heading).lower()

    penalty = 0.0

    # Non-sexual question should not return sexual-specific law
    if (
        not is_sexual_query(query)
        and any(
            term in heading_lower
            for term in [
                "rape",
                "sexual assault",
                "gang rape",
            ]
        )
    ):
        if is_homicide_query(query):
            penalty += 1.0

    # Non-negligence question should not prioritize
    # death-by-negligence provision
    if (
        is_homicide_query(query)
        and not is_negligence_query(query)
        and (
            "negligence" in heading_lower
            or "negligent" in heading_lower
        )
    ):
        penalty += 0.45

    # Section 66 should not appear for ordinary homicide
    if (
        clean_text(section).lower() == "66"
        and is_homicide_query(query)
        and not is_sexual_query(query)
    ):
        penalty += 1.0

    return min(penalty, 1.0)


# ============================================================
# CURRENT LAW BOOST
# ============================================================

def current_law_boost(
    query: str,
    act_name: str
) -> float:

    act = clean_text(act_name).lower()

    if is_explicit_ipc_query(query):

        if (
            "indian penal code" in act
            or re.search(r"\bipc\b", act)
        ):
            return 0.35

        return -0.15

    if is_explicit_bns_query(query):

        if (
            "bharatiya nyaya sanhita" in act
            or re.search(r"\bbns\b", act)
        ):
            return 0.35

        return 0.0

    # For unspecified modern Indian criminal-law queries,
    # slightly prefer BNS over old IPC.
    if (
        "bharatiya nyaya sanhita" in act
        or re.search(r"\bbns\b", act)
    ):
        return 0.15

    if (
        "indian penal code" in act
        or re.search(r"\bipc\b", act)
    ):
        return -0.10

    return 0.0


# ============================================================
# HOMOCIDE BOOST
# ============================================================

def homicide_boost(
    query: str,
    act_name: str,
    section: str,
    heading: str
) -> float:

    if not is_homicide_query(query):
        return 0.0

    section = clean_text(section).lower()
    heading = clean_text(heading).lower()
    act = clean_text(act_name).lower()

    score = 0.0

    # --------------------------------------------------------
    # Direct BNS homicide provisions
    # --------------------------------------------------------

    # Section 100 - Culpable homicide
    if section == "100":
        score += 1.20

    # Section 101 - Murder
    elif section == "101":
        score += 1.20

    # Section 103 - Punishment for murder
    elif section == "103":
        score += 1.10

    # Section 105 - Punishment for culpable homicide
    elif section == "105":
        score += 1.10

    # Section 106 - Causing death by negligence
    elif section == "106":

        if is_negligence_query(query):
            score += 1.10

        else:
            score -= 0.30

    # --------------------------------------------------------
    # Heading-based homicide relevance
    # --------------------------------------------------------

    if "culpable homicide" in heading:
        score += 0.70

    if "murder" in heading:
        score += 0.70

    if "causing death by negligence" in heading:

        if is_negligence_query(query):
            score += 0.70

        else:
            score -= 0.40

    # --------------------------------------------------------
    # Prefer BNS for unspecified modern Indian law queries
    # --------------------------------------------------------

    if (
        "bharatiya nyaya sanhita" in act
        or re.search(r"\bbns\b", act)
    ):
        score += 0.15

    return score


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    k: int = FINAL_K
) -> List[Dict[str, Any]]:

    query = clean_text(query)

    # --------------------------------------------------------
    # Prevent meaningless queries from returning random laws
    # --------------------------------------------------------

    if is_low_information_query(query):
        return []

    query_embedding = create_query_embedding(query)

    distances, indices = index.search(
        query_embedding,
        RETRIEVAL_K
    )

    candidates = []

    requested_section = extract_requested_section(query)

    for semantic_score, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        semantic_score = float(semantic_score)

        if semantic_score < MIN_RELEVANCE:
            continue

        raw_item = metadata[idx]

        act_name = get_metadata_value(
            raw_item,
            "act_name",
            "act",
            "Act",
            "Act_Name",
            default="Unknown Act"
        )

        section = get_metadata_value(
            raw_item,
            "section",
            "Section",
            "section_number",
            "section_id",
            default="Unknown"
        )

        heading = get_metadata_value(
            raw_item,
            "heading",
            "Heading",
            "title",
            "section_heading",
            default=""
        )

        text = get_text_value(raw_item)

        source_url = get_metadata_value(
            raw_item,
            "source_url",
            "source",
            "url",
            "URL",
            default=""
        )

        searchable_text = " ".join(
            [
                act_name,
                section,
                heading,
                text
            ]
        )

        # ----------------------------------------------------
        # Individual relevance scores
        # ----------------------------------------------------

        lexical = lexical_score(
            query,
            searchable_text
        )

        phrase = phrase_score(
            query,
            searchable_text
        )

        topic = legal_topic_boost(
            query,
            heading,
            text
        )

        offence = offence_match_score(
            query,
            heading,
            text
        )

        penalty = unrelated_penalty(
            query,
            searchable_text
        )

        punishment_boost = punishment_relevance_score(
            query,
            section,
            heading,
            text
        )

        punishment_penalty = punishment_mismatch_penalty(
            query,
            section,
            heading,
            text
        )

        law_boost = current_law_boost(
            query,
            act_name
        )

        homicide = homicide_boost(
            query,
            act_name,
            section,
            heading
        )

        # ----------------------------------------------------
        # Explicit section query gets a very strong boost
        # ----------------------------------------------------

        section_boost = 0.0

        if (
            requested_section
            and clean_text(section).lower()
            == requested_section
        ):
            section_boost = 1.50

        # ----------------------------------------------------
        # Final relevance score
        # ----------------------------------------------------

        final_score = (
            semantic_score * 0.50
            + lexical * 0.15
            + phrase * 0.20
            + topic * 0.10
            + offence * 0.25
            + punishment_boost * 0.20
            + law_boost
            + homicide
            + section_boost
            - penalty
            - punishment_penalty
        )

        # ----------------------------------------------------
        # Hard exclusions
        # ----------------------------------------------------

        query_sexual = is_sexual_query(query)
        query_homicide = is_homicide_query(query)
        query_negligence = is_negligence_query(query)

        heading_lower = clean_text(
            heading
        ).lower()

        section_lower = clean_text(
            section
        ).lower()

        # ----------------------------------------------------
        # Section 66 exclusion for ordinary homicide
        # ----------------------------------------------------

        if (
            section_lower == "66"
            and query_homicide
            and not query_sexual
        ):
            continue

        # ----------------------------------------------------
        # Sexual-specific provision should not be used
        # for clearly non-sexual homicide
        # ----------------------------------------------------

        if (
            query_homicide
            and not query_sexual
            and any(
                term in heading_lower
                for term in [
                    "rape",
                    "sexual assault",
                    "gang rape",
                ]
            )
        ):
            continue

        # ----------------------------------------------------
        # Death-by-negligence provision should not dominate
        # intentional-violence homicide queries
        # ----------------------------------------------------

        if (
            query_homicide
            and not query_negligence
            and (
                "negligence" in heading_lower
                or "negligent" in heading_lower
            )
        ):
            final_score -= 0.40

        # ----------------------------------------------------
        # STRONG CORE HOMICIDE RANKING
        # ----------------------------------------------------

        if query_homicide:

            core_homicide_sections = {
                "100",
                "101",
                "103",
                "105",
                "106",
            }

            if section_lower in core_homicide_sections:

                # Section 106 should not outrank intentional
                # homicide provisions unless negligence exists.
                if (
                    section_lower == "106"
                    and not query_negligence
                ):
                    final_score -= 0.25

                else:
                    final_score += 0.75

            # Secondary injury / assault provisions should not
            # outrank core homicide provisions.
            elif section_lower in {
                "122",
                "131",
            }:

                final_score -= 0.35

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        candidates.append(
            {
                "index": int(idx),

                "semantic_score": semantic_score,

                "lexical_score": float(
                    lexical
                ),

                "phrase_score": float(
                    phrase
                ),

                "topic_boost": float(
                    topic
                ),

                "offence_boost": float(
                    offence
                ),

                "law_boost": float(
                    law_boost
                ),

                "punishment_boost": float(
                    punishment_boost
                ),

                "homicide_boost": float(
                    homicide
                ),

                "penalty": float(
                    penalty
                ),

                "punishment_penalty": float(
                    punishment_penalty
                ),

                "section_boost": float(
                    section_boost
                ),

                "score": float(
                    final_score
                ),

                "act_name": act_name,

                "act": act_name,

                "section": section,

                "heading": heading,

                "text": text,

                "chunk": text,

                "source_url": source_url,

                "source": source_url,
            }
        )

    # --------------------------------------------------------
    # Sort by legal relevance
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Final filtering + deduplication
    # --------------------------------------------------------

    results = []

    seen = set()

    for item in candidates:

        score = float(
            item.get(
                "score",
                0.0
            )
        )

        # Explicit section query can pass based on
        # section match even if overall score is low.
        if (
            score < MIN_FINAL_SCORE
            and not (
                requested_section
                and clean_text(
                    item.get(
                        "section",
                        ""
                    )
                ).lower()
                == requested_section
            )
        ):
            continue

        key = (
            clean_text(
                item.get(
                    "act_name",
                    ""
                )
            ).lower(),

            clean_text(
                item.get(
                    "section",
                    ""
                )
            ).lower()
        )

        if key in seen:
            continue

        seen.add(key)

        results.append(item)

        if len(results) >= k:
            break

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: List[Dict[str, Any]]
) -> str:

    if not results:

        return (
            "NO SUFFICIENTLY RELEVANT LEGAL "
            "PROVISION WAS RETRIEVED."
        )

    blocks = []

    for number, item in enumerate(
        results,
        1
    ):

        act = clean_text(
            item.get(
                "act_name",
                item.get(
                    "act",
                    ""
                )
            )
        )

        section = clean_text(
            item.get(
                "section",
                ""
            )
        )

        heading = clean_text(
            item.get(
                "heading",
                ""
            )
        )

        text = clean_text(
            item.get(
                "text",
                item.get(
                    "chunk",
                    ""
                )
            )
        )

        source = clean_text(
            item.get(
                "source_url",
                item.get(
                    "source",
                    ""
                )
            )
        )

        score = item.get(
            "score",
            0
        )

        blocks.append(
            f"""
PROVISION {number}

Relevance Score: {score:.4f}

Act: {act}

Section: {section}

Heading: {heading}

Legal Text:
{text}

Source:
{source}
""".strip()
        )

    return "\n\n".join(blocks)


# ============================================================
# ACT NORMALIZATION
# ============================================================

def normalize_act_name(
    act_name: str
) -> str:

    act_name = clean_text(
        act_name
    )

    if not act_name:
        return "Indian Legal Provision"

    lower = act_name.lower()

    if "bharatiya nyaya sanhita" in lower:

        return (
            "The Bharatiya Nyaya Sanhita, 2023"
        )

    if lower in [
        "bns",
        "bns, 2023",
    ]:

        return (
            "The Bharatiya Nyaya Sanhita, 2023"
        )

    return act_name


# ============================================================
# SOURCE URL
# ============================================================

def get_source_url(
    item: Dict[str, Any]
) -> str:

    return clean_text(
        item.get(
            "source_url",
            item.get(
                "source",
                ""
            )
        )
    )


# ============================================================
# FIND RETRIEVED PROVISION
# ============================================================

def find_retrieved_provision(
    results: List[Dict[str, Any]],
    act: str,
    section: str
) -> Optional[Dict[str, Any]]:

    target_act = clean_text(
        act
    ).lower()

    target_section = clean_text(
        section
    ).lower()

    for item in results:

        item_act = clean_text(
            item.get(
                "act_name",
                item.get(
                    "act",
                    ""
                )
            )
        ).lower()

        item_section = clean_text(
            item.get(
                "section",
                ""
            )
        ).lower()

        if item_section != target_section:
            continue

        if not target_act:
            return item

        if (
            target_act in item_act
            or item_act in target_act
        ):
            return item

        if (
            "bharatiya nyaya sanhita" in target_act
            and
            "bharatiya nyaya sanhita" in item_act
        ):
            return item

        if (
            "indian penal code" in target_act
            and
            "indian penal code" in item_act
        ):
            return item

    return None


# ============================================================
# PUNISHMENT EXTRACTION
# ============================================================

def extract_punishment_from_text(
    text: str,
    heading: str = "",
    section: str = ""
) -> str:

    text = clean_text(text)
    heading = clean_text(heading)
    section = clean_text(section)

    if not text:
        return ""

    # --------------------------------------------------------
    # IMPORTANT:
    # These sections are definitions/substantive provisions
    # and must not inherit punishment from another section.
    # --------------------------------------------------------

    if section.lower() in {
        "100",
        "101",
    }:
        return ""

    heading_lower = heading.lower()

    has_punishment_heading = any(
        term in heading_lower
        for term in [
            "punishment",
            "punishable",
            "penalty",
        ]
    )

    has_punishment_text = bool(
        re.search(
            r"\bshall\s+be\s+punished\b",
            text,
            re.IGNORECASE
        )
        or
        re.search(
            r"\bshall\s+be\s+punishable\b",
            text,
            re.IGNORECASE
        )
    )

    if not (
        has_punishment_heading
        or has_punishment_text
    ):
        return ""

    patterns = [

        r"shall be punished with (.+?)(?:\.\s|$)",

        r"shall be punishable with (.+?)(?:\.\s|$)",

        r"punished with (.+?)(?:\.\s|$)",

        r"punishable with (.+?)(?:\.\s|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            punishment = clean_text(
                match.group(1)
            )

            if punishment:

                return (
                    punishment.rstrip(".")
                    + "."
                )

    # Fallback: return the sentence containing punishment
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    for sentence in sentences:

        lower = sentence.lower()

        if (
            "shall be punished" in lower
            or
            "shall be punishable" in lower
        ):

            return clean_text(
                sentence
            )

    return ""
# ============================================================
# VALIDATE APPLICABLE LAW
# ============================================================

def validate_applicable_law(
    law: Dict[str, Any],
    results: List[Dict[str, Any]],
    query: str
) -> Optional[Dict[str, Any]]:

    if not isinstance(
        law,
        dict
    ):
        return None

    act = normalize_act_name(
        law.get(
            "act",
            law.get(
                "act_name",
                ""
            )
        )
    )

    section = clean_text(
        law.get(
            "section",
            ""
        )
    )

    if not section:
        return None

    matched = find_retrieved_provision(
        results,
        act,
        section
    )

    # LLM cannot invent a section that was not retrieved.
    if matched is None:
        return None

    retrieved_act = normalize_act_name(
        matched.get(
            "act_name",
            matched.get(
                "act",
                act
            )
        )
    )

    heading = clean_text(
        matched.get(
            "heading",
            law.get(
                "heading",
                ""
            )
        )
    )

    text = clean_text(
        matched.get(
            "text",
            matched.get(
                "chunk",
                ""
            )
        )
    )

    source_url = get_source_url(
        matched
    )

    punishment = extract_punishment_from_text(
        text,
        heading,
        section
    )

    # --------------------------------------------------------
    # Section 66 should not be returned for ordinary homicide.
    # --------------------------------------------------------

    if (
        clean_text(section).lower() == "66"
        and is_homicide_query(query)
        and not is_sexual_query(query)
    ):
        return None

    # --------------------------------------------------------
    # Sexual-specific provision should not be returned
    # for a non-sexual homicide query.
    # --------------------------------------------------------

    if (
        is_homicide_query(query)
        and not is_sexual_query(query)
        and any(
            term in heading.lower()
            for term in [
                "rape",
                "sexual assault",
                "gang rape",
            ]
        )
    ):
        return None

    return {
        "act": retrieved_act,

        "section": section,

        "heading": heading,

        "why_relevant": clean_text(
            law.get(
                "why_relevant",
                ""
            )
        ),

        "punishment": punishment,

        "source_url": source_url,
    }


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json_from_response(
    response_text: str
) -> Dict[str, Any]:

    if response_text is None:
        return {}

    response_text = str(
        response_text
    ).strip()

    if not response_text:
        return {}

    # Remove markdown fences
    response_text = re.sub(
        r"```json\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"```",
        "",
        response_text
    )

    response_text = response_text.strip()

    # Direct JSON
    try:

        parsed = json.loads(
            response_text
        )

        if isinstance(
            parsed,
            dict
        ):
            return parsed

    except Exception:
        pass

    # Find JSON object inside text
    start = response_text.find(
        "{"
    )

    end = response_text.rfind(
        "}"
    )

    if (
        start >= 0
        and end > start
    ):

        candidate = response_text[
            start:end + 1
        ]

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                dict
            ):
                return parsed

        except Exception:
            pass

    return {}


# ============================================================
# CASE SUMMARY
# ============================================================

def build_case_summary(
    query: str
) -> str:

    query = clean_text(
        query
    )

    if not query:
        return ""

    # If it is just a legal question, don't invent facts.
    if (
        "?" in query
        and len(
            extract_key_facts(
                query
            )
        ) == 0
    ):

        return (
            "The query asks for information about "
            "a potentially relevant Indian legal provision."
        )

    if is_homicide_query(query):

        return (
            "The case describes alleged conduct involving "
            "the death of a person and raises a potentially "
            "relevant criminal-law issue."
        )

    return (
        "The query describes a legal issue and asks "
        "which Indian legal provisions may be relevant."
    )


# ============================================================
# KEY FACT EXTRACTION
# ============================================================

def extract_key_facts(
    query: str
) -> List[str]:

    query = clean_text(
        query
    )

    if not query:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        query
    )

    facts = []

    for sentence in sentences:

        sentence = clean_text(
            sentence
        )

        if not sentence:
            continue

        # Don't treat a question as a fact.
        if sentence.endswith("?"):
            continue

        lower = sentence.lower()

        # Generic UI/question phrases
        if lower in {
            "what would you like to know?",
            "what would you like to know",
            "ask a question",
            "legal query",
            "legal question",
        }:
            continue

        facts.append(
            sentence
        )

    return facts[:8]


# ============================================================
# LEGAL ISSUE FALLBACK
# ============================================================

def build_legal_issues(
    query: str,
    results: List[Dict[str, Any]]
) -> List[str]:

    query = clean_text(
        query
    )

    if not query:
        return []

    # --------------------------------------------------------
    # HOMICIDE
    # --------------------------------------------------------

    if is_homicide_query(query):

        if is_negligence_query(query):

            return [
                (
                    "Whether the alleged conduct may involve "
                    "an offence relating to death caused by "
                    "a rash or negligent act."
                )
            ]

        return [
            (
                "Whether the alleged conduct causing death "
                "may fall within murder or another offence "
                "involving culpable homicide, depending on "
                "the facts and applicable legal requirements."
            )
        ]

    # --------------------------------------------------------
    # Retrieved provision
    # --------------------------------------------------------

    if results:

        top = results[0]

        act = normalize_act_name(
            top.get(
                "act_name",
                top.get(
                    "act",
                    ""
                )
            )
        )

        section = clean_text(
            top.get(
                "section",
                ""
            )
        )

        heading = clean_text(
            top.get(
                "heading",
                ""
            )
        )

        issue = (
            f"Whether the facts or legal question "
            f"may fall within {act}, "
            f"Section {section}"
        )

        if heading:

            issue += (
                f" ({heading})"
            )

        issue += "."

        return [
            issue
        ]

    return []


# ============================================================
# LEGAL ISSUE SANITIZATION
# ============================================================

def sanitize_legal_issue(
    issue: str
) -> str:

    issue = clean_text(
        issue
    )

    if not issue:
        return ""

    lower = issue.lower()

    unsafe = [
        "is the accused guilty",
        "whether the accused is guilty",
        "is the defendant guilty",
        "whether the defendant is guilty",
        "did the accused commit",
        "has the accused committed",
    ]

    if any(
        phrase in lower
        for phrase in unsafe
    ):

        return (
            "Whether the alleged conduct may constitute "
            "the relevant offence, depending on the facts "
            "and applicable legal requirements."
        )

    return issue


# ============================================================
# FALLBACK ANALYSIS
# ============================================================

def build_fallback_analysis(
    query: str,
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    applicable_laws = []

    for item in results:

        act = normalize_act_name(
            item.get(
                "act_name",
                item.get(
                    "act",
                    ""
                )
            )
        )

        section = clean_text(
            item.get(
                "section",
                ""
            )
        )

        heading = clean_text(
            item.get(
                "heading",
                ""
            )
        )

        if not section:
            continue

        validated = validate_applicable_law(
            {
                "act": act,
                "section": section,
                "heading": heading,
            },
            results,
            query
        )

        if validated is None:
            continue

        applicable_laws.append(
            validated
        )

        if len(
            applicable_laws
        ) >= 3:
            break

    return {
        "case_summary":
            build_case_summary(
                query
            ),

        "legal_issues":
            build_legal_issues(
                query,
                results
            ),

        "key_facts":
            extract_key_facts(
                query
            ),

        "applicable_laws":
            applicable_laws,

        "current_law_note":
            "",

        "overall_explanation":
            "",
    }


# ============================================================
# STRUCTURED PROMPT
# ============================================================

def create_structured_analysis_prompt(
    query: str,
    results: List[Dict[str, Any]]
) -> str:

    context = build_context(
        results
    )

    return f"""
You are CASETag, an Indian legal information retrieval
assistant.

Your task is to explain the user's legal question using
ONLY the retrieved legal provisions.

USER QUERY:
{query}

RETRIEVED LEGAL PROVISIONS:
{context}

STRICT RULES:

1. Use ONLY the retrieved provisions.

2. Never invent an Act, Section, heading, punishment,
   source URL, or legal provision.

3. Never invent case facts.

4. If the user asks a simple legal question, describe
   the question itself. Do not create a fictional case.

5. Do not determine guilt or innocence.

6. Use neutral wording:
   "may be relevant",
   "may apply depending on the facts",
   "potentially applicable".

7. A punishment must belong to the SAME section from
   which it was retrieved.

8. Do not copy punishment from another section.

9. Section 100 of BNS must not receive Section 105
   punishment.

10. Section 101 must not receive Section 103 punishment.

11. If the retrieved section itself does not contain
    punishment, return an empty punishment string.

12. Do not use a sexual-offence-specific provision merely
    because another provision contains the word "death".

13. For a homicide query, distinguish between murder,
    culpable homicide and death caused by negligence
    based on the retrieved legal text and facts.

14. Do not claim that a person definitely committed
    an offence.

15. Keep the answer concise and structured.

16. Return ONLY valid JSON.

Use exactly this structure:

{{
  "case_summary": "...",
  "legal_issues": [
    "..."
  ],
  "key_facts": [
    "..."
  ],
  "applicable_laws": [
    {{
      "act": "...",
      "section": "...",
      "heading": "...",
      "why_relevant": "...",
      "punishment": "...",
      "source_url": "..."
    }}
  ],
  "current_law_note": "",
  "overall_explanation": ""
}}
""".strip()


# ============================================================
# STRUCTURED ANALYSIS
# ============================================================

def generate_structured_analysis(
    query: str,
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    fallback = build_fallback_analysis(
        query,
        results
    )

    if not results:
        return fallback

    prompt = create_structured_analysis_prompt(
        query,
        results
    )

    try:

        response = llm.invoke(
            prompt
        )

        if hasattr(
            response,
            "content"
        ):

            response_text = response.content

        else:

            response_text = str(
                response
            )

        parsed = extract_json_from_response(
            response_text
        )

        if not parsed:
            return fallback

        # ----------------------------------------------------
        # CASE SUMMARY
        # ----------------------------------------------------

        case_summary = clean_text(
            parsed.get(
                "case_summary",
                ""
            )
        )

        if not case_summary:

            case_summary = fallback[
                "case_summary"
            ]

        # ----------------------------------------------------
        # LEGAL ISSUES
        # ----------------------------------------------------

        legal_issues = []

        raw_issues = parsed.get(
            "legal_issues",
            []
        )

        if isinstance(
            raw_issues,
            list
        ):

            for issue in raw_issues:

                issue = sanitize_legal_issue(
                    issue
                )

                if issue:

                    legal_issues.append(
                        issue
                    )

        if not legal_issues:

            legal_issues = fallback[
                "legal_issues"
            ]

        # ----------------------------------------------------
        # KEY FACTS
        # ----------------------------------------------------

        key_facts = []

        raw_facts = parsed.get(
            "key_facts",
            []
        )

        if isinstance(
            raw_facts,
            list
        ):

            for fact in raw_facts:

                fact = clean_text(
                    fact
                )

                if fact:

                    key_facts.append(
                        fact
                    )

        # IMPORTANT:
        # Don't allow Groq to invent facts for a simple query.
        fallback_facts = fallback[
            "key_facts"
        ]

        if fallback_facts:

            key_facts = fallback_facts

        elif not key_facts:

            key_facts = []

        # ----------------------------------------------------
        # APPLICABLE LAWS
        # ----------------------------------------------------

        applicable_laws = []

        raw_laws = parsed.get(
            "applicable_laws",
            []
        )

        if isinstance(
            raw_laws,
            list
        ):

            seen = set()

            for law in raw_laws:

                if not isinstance(
                    law,
                    dict
                ):
                    continue

                validated = validate_applicable_law(
                    law,
                    results,
                    query
                )

                if validated is None:
                    continue

                key = (
                    clean_text(
                        validated.get(
                            "act",
                            ""
                        )
                    ).lower(),

                    clean_text(
                        validated.get(
                            "section",
                            ""
                        )
                    ).lower(),
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                applicable_laws.append(
                    validated
                )

        # ----------------------------------------------------
        # FALLBACK IF GROQ RETURNED NOTHING VALID
        # ----------------------------------------------------

        if not applicable_laws:

            applicable_laws = fallback[
                "applicable_laws"
            ]

        # ----------------------------------------------------
        # SORT BY RETRIEVAL SCORE
        # ----------------------------------------------------

        def law_score(
            law: Dict[str, Any]
        ) -> float:

            matched = find_retrieved_provision(
                results,
                law.get(
                    "act",
                    ""
                ),
                law.get(
                    "section",
                    ""
                )
            )

            if matched:

                return float(
                    matched.get(
                        "score",
                        0.0
                    )
                )

            return -1.0

        applicable_laws.sort(
            key=law_score,
            reverse=True
        )

        # Keep output focused
        applicable_laws = (
            applicable_laws[:3]
        )

        # ----------------------------------------------------
        # FINAL PUNISHMENT SAFETY
        # ----------------------------------------------------

        for law in applicable_laws:

            section = clean_text(
                law.get(
                    "section",
                    ""
                )
            )

            if section in {
                "100",
                "101",
            }:

                law[
                    "punishment"
                ] = ""

        return {
            "case_summary":
                case_summary,

            "legal_issues":
                legal_issues,

            "key_facts":
                key_facts,

            "applicable_laws":
                applicable_laws,

            "current_law_note":
                "",

            "overall_explanation":
                "",
        }

    except Exception as error:

        print(
            "Structured analysis generation error:",
            repr(error)
        )

        return fallback


# ============================================================
# MAIN QUERY PIPELINE
# ============================================================

def answer_query(
    query: str,
    k: int = FINAL_K
) -> Dict[str, Any]:

    query = clean_text(
        query
    )

    if not query:

        return {
            "query": "",

            "results": [],

            "context": "",

            "answer": (
                "Please provide a legal query "
                "or case description."
            ),

            "analysis": {
                "case_summary": "",

                "legal_issues": [],

                "key_facts": [],

                "applicable_laws": [],

                "current_law_note": "",

                "overall_explanation": "",
            },
        }

    results = retrieve(
        query,
        k=k
    )

    context = build_context(
        results
    )

    # ONE LLM CALL ONLY
    analysis = generate_structured_analysis(
        query,
        results
    )

    return {
        "query": query,

        "results": results,

        "context": context,

        "answer": "",

        "analysis": analysis,
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def query_rag(
    query: str,
    k: int = FINAL_K
) -> Dict[str, Any]:

    return answer_query(
        query,
        k=k
    )


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(
    query: str,
    k: int = FINAL_K
) -> None:

    print()
    print("=" * 80)
    print("CASETAG RETRIEVAL DEBUG")
    print("=" * 80)

    print()
    print("QUERY:")
    print(query)

    results = retrieve(
        query,
        k=k
    )

    if not results:

        print()
        print(
            "No relevant legal provisions found."
        )

        return

    for number, item in enumerate(
        results,
        1
    ):

        print()
        print(
            f"RESULT {number}"
        )

        print(
            "-" * 60
        )

        print(
            "Score:",
            round(
                item.get(
                    "score",
                    0
                ),
                4
            )
        )

        print(
            "Semantic:",
            round(
                item.get(
                    "semantic_score",
                    0
                ),
                4
            )
        )

        print(
            "Act:",
            item.get(
                "act_name",
                ""
            )
        )

        print(
            "Section:",
            item.get(
                "section",
                ""
            )
        )

        print(
            "Heading:",
            item.get(
                "heading",
                ""
            )
        )

        print(
            "Source:",
            item.get(
                "source_url",
                ""
            )
        )

        text = clean_text(
            item.get(
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
            "Text:",
            text
        )

    print()
    print("=" * 80)


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_queries = [

        "What is the punishment for theft under Indian law?",

        "What is the punishment for rape under BNS?",

        (
            "The accused allegedly attacked the victim "
            "with a sharp object and the victim later died. "
            "Which legal provision may be relevant?"
        ),
    ]

    for query in test_queries:

        print()
        print("=" * 80)
        print(
            "TEST QUERY:",
            query
        )
        print("=" * 80)

        try:

            response = answer_query(
                query
            )

            print()

            print(
                json.dumps(
                    response.get(
                        "analysis",
                        {}
                    ),
                    indent=2,
                    ensure_ascii=False
                )
            )

        except Exception as error:

            print(
                "CASETAG test failed:",
                repr(error)
            )