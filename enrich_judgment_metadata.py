import os
import re
import pickle
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PARQUET_PATH = "data/criminal_judgments.parquet"

METADATA_PATH = (
    "vectorstore/judgments/judgments_metadata.pkl"
)

OUTPUT_PATH = (
    "vectorstore/judgments/"
    "judgments_metadata_enriched.pkl"
)


# ============================================================
# EXTRACT FILE NUMBER
# ============================================================

def get_file_number(filename):

    if not filename:
        return None

    filename = os.path.basename(
        str(filename)
    )

    match = re.match(
        r"^(\d+)_",
        filename
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CASETAG - ENRICH JUDGMENT METADATA")
print("=" * 70)


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("\nLoading criminal judgments metadata...")

df = pd.read_parquet(
    PARQUET_PATH
)

print(
    f"Total source judgments: {len(df)}"
)


# ============================================================
# LOAD EXISTING VECTORSTORE METADATA
# ============================================================

print("\nLoading judgment vectorstore metadata...")

with open(
    METADATA_PATH,
    "rb"
) as f:

    metadata = pickle.load(f)


print(
    f"Existing metadata records: {len(metadata)}"
)


# ============================================================
# ENRICH
# ============================================================

enriched = []

matched = 0
unmatched = 0


for item in metadata:

    item = dict(item)

    judgment_file = item.get(
        "judgment_file",
        ""
    )

    file_number = get_file_number(
        judgment_file
    )

    # --------------------------------------------------------
    # Match 001 -> first dataframe row
    # Match 002 -> second dataframe row
    # --------------------------------------------------------

    if (
        file_number is not None
        and
        1 <= file_number <= len(df)
    ):

        row = df.iloc[
            file_number - 1
        ]

        item.update({

            "case_title":
                row.get(
                    "case_title",
                    ""
                ),

            "court_name":
                row.get(
                    "court_name",
                    ""
                ),

            "decision_date":
                row.get(
                    "decision_date",
                    ""
                ),

            "docket_number":
                row.get(
                    "docket_number",
                    ""
                ),

            "neutral_citation":
                row.get(
                    "neutral_citation",
                    ""
                ),

            "law_report_citation":
                row.get(
                    "law_report_citation",
                    ""
                ),

            "disposition_text":
                row.get(
                    "disposition_text",
                    ""
                ),

            "source_pdf_s3_url":
                row.get(
                    "source_pdf_s3_url",
                    ""
                ),

            "court_code":
                row.get(
                    "court_code",
                    ""
                ),

            "bench_name":
                row.get(
                    "bench_name",
                    ""
                ),

            "presiding_judge":
                row.get(
                    "presiding_judge",
                    ""
                ),

            "source_filename":
                row.get(
                    "source_filename",
                    ""
                ),
        })

        matched += 1

    else:

        unmatched += 1

        item.update({

            "case_title": "",

            "court_name": "",

            "decision_date": "",

            "docket_number": "",

            "neutral_citation": "",

            "law_report_citation": "",

            "disposition_text": "",

            "source_pdf_s3_url": "",

            "court_code": "",

            "bench_name": "",

            "presiding_judge": "",

            "source_filename": "",
        })

    enriched.append(
        item
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "wb"
) as f:

    pickle.dump(
        enriched,
        f
    )


# ============================================================
# VERIFY FIRST RECORD
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

first = enriched[0]

print(
    "\nJudgment file:"
)

print(
    first.get(
        "judgment_file",
        ""
    )
)

print(
    "\nCase:"
)

print(
    first.get(
        "case_title",
        ""
    )
)

print(
    "\nCourt:"
)

print(
    first.get(
        "court_name",
        ""
    )
)

print(
    "\nDocket:"
)

print(
    first.get(
        "docket_number",
        ""
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ENRICHMENT COMPLETE")
print("=" * 70)

print(
    f"\nOriginal metadata : {len(metadata)}"
)

print(
    f"Matched           : {matched}"
)

print(
    f"Unmatched         : {unmatched}"
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 70)