import pandas as pd
import requests
import os
import re
import time
from pypdf import PdfReader

INPUT_FILE = "data/criminal_judgments.parquet"
OUTPUT_DIR = "judgment_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CASETAG - Download All Criminal Judgments")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nTotal criminal judgments: {len(df)}")

successful = 0
failed = 0
skipped = 0

for i, (_, row) in enumerate(df.iterrows(), start=1):

    case_title = str(row.get("case_title", "Unknown Case"))
    court = str(row.get("court_name", "Unknown Court"))
    date = str(row.get("decision_date", "Unknown Date"))
    pdf_url = row.get("source_pdf_s3_url")

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        case_title
    )[:80]

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{i:03d}_{safe_name}.pdf"
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"{i:03d}_{safe_name}.txt"
    )

    print("\n" + "-" * 70)
    print(f"[{i}/{len(df)}] {case_title}")
    print(f"Court: {court}")
    print(f"Date : {date}")

    # Already downloaded
    if os.path.exists(txt_path):
        print("SKIPPED - already downloaded")
        skipped += 1
        continue

    if pd.isna(pdf_url):
        print("SKIPPED - no PDF URL")
        skipped += 1
        continue

    try:

        print("Downloading PDF...")

        response = requests.get(
            pdf_url,
            timeout=60
        )

        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print(
            f"Downloaded: {len(response.content):,} bytes"
        )

        # Extract PDF text
        reader = PdfReader(pdf_path)

        text_parts = []

        for page in reader.pages:

            try:
                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

            except Exception as e:
                print(
                    f"Warning: Could not extract one page: {e}"
                )

        full_text = "\n".join(text_parts).strip()

        # Save text
        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(full_text)

        print(f"Pages     : {len(reader.pages)}")
        print(f"Characters: {len(full_text):,}")

        if len(full_text) < 300:

            print(
                "WARNING - Very little text extracted"
            )

        else:

            successful += 1
            print("SUCCESS")

        # Small delay
        time.sleep(0.5)

    except Exception as e:

        failed += 1

        print(
            f"FAILED: {type(e).__name__}: {e}"
        )

        # Remove incomplete files
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except:
                pass

        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
            except:
                pass


print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(f"Total records : {len(df)}")
print(f"Successful    : {successful}")
print(f"Skipped       : {skipped}")
print(f"Failed        : {failed}")

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\nNext step:")
print("Build the Previous Judgments FAISS vector store.")