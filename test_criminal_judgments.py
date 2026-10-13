import pandas as pd
import requests
import os
import re
from pypdf import PdfReader

# --------------------------------------------------
# FILES
# --------------------------------------------------

INPUT_FILE = "data/criminal_judgments.parquet"
OUTPUT_DIR = "judgment_test_criminal"

TEST_COUNT = 20


# --------------------------------------------------
# SETUP
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_FILE)

print("=" * 70)
print("CASETAG - Criminal Judgment PDF Test")
print("=" * 70)

print(f"\nTotal criminal judgments: {len(df)}")

# Take first 20 records having PDF URLs
df = df[
    df["source_pdf_s3_url"].notna()
].head(TEST_COUNT)

print(f"Testing: {len(df)} judgments\n")


# --------------------------------------------------
# DOWNLOAD + EXTRACT
# --------------------------------------------------

successful = 0
failed = 0

results = []

for i, (_, row) in enumerate(df.iterrows(), start=1):

    case_title = str(row.get("case_title", "Unknown Case"))
    court = str(row.get("court_name", "Unknown Court"))
    date = str(row.get("decision_date", "Unknown Date"))
    pdf_url = row.get("source_pdf_s3_url")

    print("=" * 70)
    print(f"TEST {i}")
    print("=" * 70)

    print(f"Case  : {case_title}")
    print(f"Court : {court}")
    print(f"Date  : {date}")

    if pd.isna(pdf_url):
        print("ERROR: No PDF URL")
        failed += 1
        continue

    try:

        # --------------------------------------------------
        # SAFE FILE NAME
        # --------------------------------------------------

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            case_title
        )[:100]

        pdf_path = os.path.join(
            OUTPUT_DIR,
            f"{i:03d}_{safe_name}.pdf"
        )

        txt_path = os.path.join(
            OUTPUT_DIR,
            f"{i:03d}_{safe_name}.txt"
        )

        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        response = requests.get(
            pdf_url,
            timeout=60
        )

        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # --------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------

        reader = PdfReader(pdf_path)

        text_parts = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts)

        # Save extracted text
        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(full_text)

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        char_count = len(full_text)

        print(f"Downloaded : {len(response.content)} bytes")
        print(f"Pages      : {len(reader.pages)}")
        print(f"Characters : {char_count}")

        print("\nTEXT PREVIEW:")
        print("-" * 70)

        preview = full_text[:1000]

        print(preview)

        print("-" * 70)

        if char_count > 300:
            successful += 1
        else:
            print("WARNING: Very little text extracted.")

        results.append({
            "case_title": case_title,
            "court": court,
            "date": date,
            "pages": len(reader.pages),
            "characters": char_count,
            "pdf_path": pdf_path,
            "text_path": txt_path
        })

    except Exception as e:

        failed += 1

        print(f"ERROR: {e}")


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"Total tested       : {len(df)}")
print(f"Successful         : {successful}")
print(f"Failed             : {failed}")

print(f"\nFiles saved inside:")
print(OUTPUT_DIR)

print("\nNext step:")
print("If extraction quality looks good, we will build the")
print("Previous Judgments FAISS vector store.")