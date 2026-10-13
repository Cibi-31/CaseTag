
import pandas as pd
import os


# ============================================================
# CASETAG - DATA LOADER
# Loads all 3 legal datasets
# ============================================================

DATA_DIR = "data"

ACTS_FILE = os.path.join(DATA_DIR, r"C:\Users\T9960\Downloads\casetag\data\acts.csv")
SECTIONS_FILE = os.path.join(DATA_DIR, r"C:\Users\T9960\Downloads\casetag\data\legal_sections.csv")
LEGAL_SECTIONS_FILE = os.path.join(DATA_DIR, r"C:\Users\T9960\Downloads\casetag\data\legal_sections.csv")


# ============================================================
# 1. LOAD ACTS DATASET
# ============================================================

def load_acts():
    print("\nLoading Acts dataset...")

    if not os.path.exists(ACTS_FILE):
        raise FileNotFoundError(
            f"Acts dataset not found: {ACTS_FILE}"
        )

    df_acts = pd.read_csv(
        ACTS_FILE,
        encoding="utf-8-sig"
    )

    print("Acts dataset loaded successfully.")
    print("Rows:", len(df_acts))
    print("Columns:", df_acts.columns.tolist())

    return df_acts


# ============================================================
# 2. LOAD SECTIONS DATASET
# ============================================================

def load_sections():
    print("\nLoading Sections dataset...")

    if not os.path.exists(SECTIONS_FILE):
        raise FileNotFoundError(
            f"Sections dataset not found: {SECTIONS_FILE}"
        )

    df_sections = pd.read_csv(
        SECTIONS_FILE,
        encoding="utf-8-sig"
    )

    print("Sections dataset loaded successfully.")
    print("Rows:", len(df_sections))
    print("Columns:", df_sections.columns.tolist())

    return df_sections


# ============================================================
# 3. LOAD LEGAL SECTIONS DATASET
# ============================================================

def load_legal_sections():
    print("\nLoading Legal Sections dataset...")

    if not os.path.exists(LEGAL_SECTIONS_FILE):
        raise FileNotFoundError(
            f"Legal sections dataset not found: {LEGAL_SECTIONS_FILE}"
        )

    df_legal_sections = pd.read_csv(
        LEGAL_SECTIONS_FILE,
        encoding="utf-8-sig"
    )

    print("Legal Sections dataset loaded successfully.")
    print("Rows:", len(df_legal_sections))
    print("Columns:", df_legal_sections.columns.tolist())

    return df_legal_sections


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df, name):

    print(f"\n========== {name.upper()} VALIDATION ==========")

    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("        CASETAG LEGAL DATA LOADER")
    print("=" * 60)

    # Load all three datasets
    acts = load_acts()
    sections = load_sections()
    legal_sections = load_legal_sections()

    # Validate all datasets
    validate_dataset(
        acts,
        "Acts Dataset"
    )

    validate_dataset(
        sections,
        "Sections Dataset"
    )

    validate_dataset(
        legal_sections,
        "Legal Sections Dataset"
    )

    # ========================================================
    # SAMPLE DATA
    # ========================================================

    print("\n\n========== ACTS SAMPLE ==========")
    print(acts.head(3).to_string(index=False))

    print("\n\n========== SECTIONS SAMPLE ==========")
    print(sections.head(3).to_string(index=False))

    print("\n\n========== LEGAL SECTIONS SAMPLE ==========")
    print(
        legal_sections.head(3).to_string(index=False)
    )

    print("\n")
    print("=" * 60)
    print("All three datasets loaded successfully!")
    print("=" * 60)

