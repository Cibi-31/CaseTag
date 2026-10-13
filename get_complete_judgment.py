import os
import pickle
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

METADATA_PATH = (
    "vectorstore/judgments/"
    "judgments_metadata_enriched.pkl"
)

JUDGMENT_DATA_DIR = "judgment_data"


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("CASETAG - COMPLETE JUDGMENT LOADER")
print("=" * 70)

with open(
    METADATA_PATH,
    "rb"
) as f:

    metadata = pickle.load(f)


print(
    f"\nMetadata records: {len(metadata)}"
)


# ============================================================
# GROUP CHUNKS BY JUDGMENT
# ============================================================

judgments = defaultdict(list)


for item in metadata:

    filename = item.get(
        "judgment_file",
        ""
    )

    if not filename:
        continue

    judgments[
        filename
    ].append(item)


print(
    f"Unique judgments: {len(judgments)}"
)


# ============================================================
# SORT CHUNKS
# ============================================================

for filename in judgments:

    judgments[filename].sort(
        key=lambda x:
            int(
                x.get(
                    "chunk_id",
                    0
                )
            )
    )


# ============================================================
# LOAD COMPLETE JUDGMENT
# ============================================================

def load_complete_judgment(
    judgment_file
):

    chunks = judgments.get(
        judgment_file,
        []
    )

    if not chunks:

        return None


    first = chunks[0]


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    result = {

        "judgment_file":
            judgment_file,

        "case_title":
            first.get(
                "case_title",
                ""
            ),

        "court_name":
            first.get(
                "court_name",
                ""
            ),

        "decision_date":
            first.get(
                "decision_date",
                ""
            ),

        "docket_number":
            first.get(
                "docket_number",
                ""
            ),

        "neutral_citation":
            first.get(
                "neutral_citation",
                ""
            ),

        "law_report_citation":
            first.get(
                "law_report_citation",
                ""
            ),

        "disposition_text":
            first.get(
                "disposition_text",
                ""
            ),

        "source_pdf_s3_url":
            first.get(
                "source_pdf_s3_url",
                ""
            ),

        "chunks":
            chunks,
    }


    # --------------------------------------------------------
    # Combine chunk text
    # --------------------------------------------------------

    result["full_text"] = "\n\n".join(

        chunk.get(
            "text",
            ""
        )

        for chunk in chunks

    )


    return result


# ============================================================
# FINAL ORDER DETECTION
# ============================================================

def find_order_chunks(
    judgment
):

    chunks = judgment.get(
        "chunks",
        []
    )

    if not chunks:

        return []


    order_keywords = [

        "accordingly",

        "in the result",

        "in view of the above",

        "this bail application",

        "bail application is",

        "bail is granted",

        "bail is rejected",

        "bail application stands rejected",

        "anticipatory bail",

        "application is allowed",

        "application is dismissed",

        "appeal is allowed",

        "appeal is dismissed",

        "petition is allowed",

        "petition is dismissed",

        "convicted",

        "acquitted",

        "sentence",

        "sentenced",

        "life imprisonment",

        "death penalty",

        "death sentence",

        "imprisonment",

        "fine",
    ]


    matches = []


    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )

        lower = text.lower()


        score = 0


        for keyword in order_keywords:

            if keyword in lower:

                score += 1


        if score > 0:

            matches.append({

                "chunk_id":
                    chunk.get(
                        "chunk_id",
                        ""
                    ),

                "score":
                    score,

                "text":
                    text,
            })


    matches.sort(

        key=lambda x:
            (
                x["score"],
                int(
                    x["chunk_id"]
                )
            ),

        reverse=True

    )


    return matches


# ============================================================
# INTERACTIVE TEST
# ============================================================

print("\n" + "=" * 70)
print("TYPE A JUDGMENT FILE")
print("=" * 70)

print(
    "\nExample:"
)

print(
    "001_CR__MISC__4828_2020_of_Rambabu_Prasad_"
    "Vs_The_State_of_Bihar.txt"
)

filename = input(
    "\nJudgment file: "
).strip()


# ============================================================
# LOAD
# ============================================================

judgment = load_complete_judgment(
    filename
)


if judgment is None:

    print(
        "\nJudgment not found."
    )

    raise SystemExit


# ============================================================
# DISPLAY METADATA
# ============================================================

print("\n" + "=" * 70)
print("JUDGMENT DETAILS")
print("=" * 70)

print(
    "\nCASE:"
)

print(
    judgment["case_title"]
)

print(
    "\nCOURT:"
)

print(
    judgment["court_name"]
)

print(
    "\nDECISION DATE:"
)

print(
    judgment["decision_date"]
)

print(
    "\nDOCKET:"
)

print(
    judgment["docket_number"]
)

print(
    "\nNEUTRAL CITATION:"
)

print(
    judgment["neutral_citation"]
)

print(
    "\nNUMBER OF CHUNKS:"
)

print(
    len(
        judgment["chunks"]
    )
)


# ============================================================
# FIND ORDER
# ============================================================

order_chunks = find_order_chunks(
    judgment
)


print("\n" + "=" * 70)
print("POSSIBLE FINAL / ORDER CHUNKS")
print("=" * 70)


if not order_chunks:

    print(
        "\nNo obvious order keywords found."
    )

else:

    for i, item in enumerate(
        order_chunks[:5],
        1
    ):

        print(
            f"\n#{i}"
        )

        print(
            f"Chunk: {item['chunk_id']}"
        )

        print(
            f"Order keyword score: "
            f"{item['score']}"
        )

        print(
            "\nText:"
        )

        print(
            item["text"]
        )

        print(
            "\n" + "-" * 70
        )


# ============================================================
# FULL JUDGMENT LENGTH
# ============================================================

print("\n" + "=" * 70)

print(
    f"Complete extracted judgment length: "
    f"{len(judgment['full_text'])} characters"
)

print("=" * 70)