import pandas as pd
import os
import re

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

INPUT_FILE = "data/indian_case_laws_sample_v1.parquet"
OUTPUT_FILE = "data/criminal_judgments.parquet"


# --------------------------------------------------
# CRIMINAL CASE SIGNALS
# --------------------------------------------------

CRIMINAL_PATTERNS = [
    r"\bcr\.?\s*misc\b",
    r"\bcriminal\b",
    r"\bcr\.?\s*w\.?\s*j\.?\s*c\b",
    r"\bcriminal\s*writ\b",
    r"\bcriminal\s*appeal\b",
    r"\bcriminal\s*revision\b",
    r"\bcriminal\s*petition\b",
    r"\bsessions\b",
    r"\bbail\b",
    r"\bfir\b",
    r"\bfirst\s+information\s+report\b",

    # Common criminal-law sections
    r"\b302\b",
    r"\b307\b",
    r"\b304\b",
    r"\b376\b",
    r"\b354\b",
    r"\b363\b",
    r"\b364\b",
    r"\b365\b",
    r"\b366\b",
    r"\b376\b",
    r"\b379\b",
    r"\b380\b",
    r"\b392\b",
    r"\b395\b",
    r"\b406\b",
    r"\b409\b",
    r"\b420\b",
    r"\b498a\b",
    r"\b506\b",

    # Common offences
    r"\bmurder\b",
    r"\brape\b",
    r"\btheft\b",
    r"\brobbery\b",
    r"\bdacoity\b",
    r"\bcheating\b",
    r"\bkidnapping\b",
    r"\babduction\b",
    r"\bassault\b",
    r"\bforgery\b",
]


def is_criminal_case(row):
    """
    Check whether a judgment contains signals
    suggesting that it is a criminal case.
    """

    fields = [
        row.get("case_title", ""),
        row.get("party_caption", ""),
        row.get("docket_number", ""),
        row.get("court_name", ""),
        row.get("disposition_text", ""),
        row.get("indexable_text", ""),
        row.get("headnote_text", ""),
    ]

    text = " ".join(
        str(value)
        for value in fields
        if pd.notna(value)
    ).lower()

    for pattern in CRIMINAL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CASETAG - Criminal Judgment Filter")
    print("=" * 60)

    # Check input file
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    print(f"\nLoading:")
    print(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nTotal judgments: {len(df)}")

    # Apply filter
    print("\nFiltering likely criminal judgments...")

    df["is_likely_criminal"] = df.apply(
        is_criminal_case,
        axis=1
    )

    criminal_df = df[
        df["is_likely_criminal"] == True
    ].copy()

    # Remove helper column
    criminal_df.drop(
        columns=["is_likely_criminal"],
        inplace=True
    )

    # Save
    criminal_df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FILTER RESULT")
    print("=" * 60)

    print(f"Total judgments      : {len(df)}")
    print(f"Likely criminal      : {len(criminal_df)}")
    print(
        f"Other judgments      : {len(df) - len(criminal_df)}"
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    # Show sample
    print("\nSample criminal judgments:")
    print("-" * 60)

    columns_to_show = [
        "case_title",
        "court_name",
        "decision_date",
        "disposition_text"
    ]

    available_columns = [
        col for col in columns_to_show
        if col in criminal_df.columns
    ]

    print(
        criminal_df[available_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()