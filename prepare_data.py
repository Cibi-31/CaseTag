
import pandas as pd
import os


# ============================================================
# CASETAG - KNOWLEDGE BASE PREPARATION
# ============================================================

DATA_DIR = "data"

ACTS_FILE = os.path.join(DATA_DIR, r"C:\Users\T9960\Downloads\casetag\data\acts.csv")
LEGAL_SECTIONS_FILE = os.path.join(DATA_DIR, r"C:\Users\T9960\Downloads\casetag\data\legal_sections.csv")

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    r"C:\Users\T9960\Downloads\casetag\data\casetag_knowledge_base.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("       CASETAG KNOWLEDGE BASE PREPARATION")
    print("=" * 60)

    print("\nLoading Acts dataset...")
    acts = pd.read_csv(
        ACTS_FILE,
        encoding="utf-8-sig"
    )

    print(f"Acts loaded: {len(acts)}")

    print("\nLoading Legal Sections dataset...")
    sections = pd.read_csv(
        LEGAL_SECTIONS_FILE,
        encoding="utf-8-sig"
    )

    print(f"Legal sections loaded: {len(sections)}")

    return acts, sections


# ============================================================
# CLEAN ACT DATA
# ============================================================

def clean_acts(acts):

    print("\nCleaning Acts dataset...")

    # Keep useful metadata
    acts_clean = acts[
        [
            "id",
            "short_title",
            "long_title",
            "act_year",
            "jurisdiction",
            "ministry",
            "department",
            "in_force",
            "section_count"
        ]
    ].copy()

    # Rename ID so it matches the section dataset
    acts_clean = acts_clean.rename(
        columns={
            "id": "act"
        }
    )

    # Fill missing optional metadata
    acts_clean["long_title"] = (
        acts_clean["long_title"]
        .fillna("")
    )

    acts_clean["ministry"] = (
        acts_clean["ministry"]
        .fillna("")
    )

    acts_clean["department"] = (
        acts_clean["department"]
        .fillna("")
    )

    return acts_clean


# ============================================================
# CLEAN SECTION DATA
# ============================================================

def clean_sections(sections):

    print("\nCleaning Legal Sections dataset...")

    sections_clean = sections[
        [
            "act",
            "section",
            "heading",
            "text",
            "source_url"
        ]
    ].copy()

    # Remove records without section text
    sections_clean = sections_clean.dropna(
        subset=["text"]
    )

    sections_clean = sections_clean[
        sections_clean["text"]
        .astype(str)
        .str.strip()
        != ""
    ]

    # Heading can be missing
    sections_clean["heading"] = (
        sections_clean["heading"]
        .fillna("")
    )

    # Remove exact duplicate Act + Section records
    sections_clean = sections_clean.drop_duplicates(
        subset=["act", "section"]
    )

    return sections_clean


# ============================================================
# MERGE ACT + SECTION DATA
# ============================================================

def create_knowledge_base(acts, sections):

    print("\nCombining Acts and Sections...")

    knowledge_base = sections.merge(
        acts,
        on="act",
        how="left"
    )

    # Create a clean Act name
    knowledge_base["act_name"] = (
        knowledge_base["short_title"]
        .fillna(
            knowledge_base["act"]
        )
    )

    # Reorder columns
    knowledge_base = knowledge_base[
        [
            "act",
            "act_name",
            "section",
            "heading",
            "text",
            "source_url",
            "act_year",
            "jurisdiction",
            "ministry",
            "department",
            "in_force",
            "section_count"
        ]
    ]

    return knowledge_base


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================

def save_knowledge_base(knowledge_base):

    knowledge_base.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKnowledge base saved successfully!")
    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {len(knowledge_base)}")
    print(f"Columns: {len(knowledge_base.columns)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    acts, sections = load_data()

    acts = clean_acts(acts)

    sections = clean_sections(sections)

    knowledge_base = create_knowledge_base(
        acts,
        sections
    )

    save_knowledge_base(
        knowledge_base
    )

    print("\n" + "=" * 60)
    print("          CASETAG PREPARATION COMPLETE")
    print("=" * 60)

    print("\nSample records:\n")

    print(
        knowledge_base[
            [
                "act_name",
                "section",
                "heading"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

