import json
import os

from langchain_groq import ChatGroq

from retrieve_previous_judgments import retrieve_previous_judgments


# ============================================================
# GROQ CONFIG
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

# Keep your existing API key setup here.
# Prefer environment variable:
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# CURRENT CASE
# ============================================================

CASE_QUERY = """
The accused stabbed the victim with a knife during a quarrel.
The victim died from the injuries. The accused claims that he
did not intend to cause death.
"""


# ============================================================
# GROQ
# ============================================================

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    # Remove excessive whitespace
    text = " ".join(text.split())

    return text.strip()


# ============================================================
# LIMIT TEXT SIZE
# ============================================================

def truncate_text(text, max_chars=3500):
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "... [truncated]"


# ============================================================
# BUILD GROQ CONTEXT
# ============================================================

def build_judgment_context(judgments):

    context_parts = []

    for i, judgment in enumerate(judgments, start=1):

        case_title = clean_text(
            judgment.get("case_title", "Not available")
        )

        court_name = clean_text(
            judgment.get("court_name", "Not available")
        )

        decision_date = clean_text(
            judgment.get("decision_date", "Not available")
        )

        docket_number = clean_text(
            judgment.get("docket_number", "Not available")
        )

        similarity = judgment.get("similarity", "")

        matched_text = truncate_text(
            judgment.get("matched_text", ""),
            2200
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Court order is the strongest evidence.
        # ----------------------------------------------------

        order_chunks = judgment.get("order_chunks", [])

        order_texts = []

        for order_chunk in order_chunks:

            if isinstance(order_chunk, dict):

                chunk_text = (
                    order_chunk.get("text")
                    or order_chunk.get("chunk_text")
                    or order_chunk.get("content")
                    or ""
                )

            else:
                chunk_text = str(order_chunk)

            chunk_text = clean_text(chunk_text)

            if chunk_text:
                order_texts.append(chunk_text)

        order_text = "\n\n".join(order_texts)

        order_text = truncate_text(
            order_text,
            3500
        )

        # ----------------------------------------------------
        # Build evidence
        # ----------------------------------------------------

        block = f"""
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

------------------------------------------------------------
COURT ORDER / FINAL DECISION
------------------------------------------------------------

{order_text if order_text else "NO EXPLICIT ORDER CHUNK FOUND"}

------------------------------------------------------------
RETRIEVED RELEVANT TEXT
------------------------------------------------------------

{matched_text if matched_text else "No matched text available"}

============================================================
"""

        context_parts.append(block)

    return "\n".join(context_parts)


# ============================================================
# GROQ PROMPT
# ============================================================

def create_prompt(case_query, judgment_context):

    return f"""
You are a legal judgment extraction assistant for CaseTag.

You are NOT deciding the current case.

Your task is to extract factual information about PREVIOUS
COURT JUDGMENTS from the supplied retrieval evidence.

CURRENT CASE:
{case_query}

PREVIOUS JUDGMENT EVIDENCE:
{judgment_context}


============================================================
CRITICAL EXTRACTION RULES
============================================================

1. COURT ORDER HAS THE HIGHEST PRIORITY.

If a section titled:

COURT ORDER / FINAL DECISION

contains an explicit court decision, you MUST extract that
decision.

DO NOT say:

"Decision not available"

when an explicit court order is present.

------------------------------------------------------------

2. USE THE ACTUAL WORDING OF THE COURT ORDER.

For example, if the order says:

"the bail application of the applicant is hereby allowed"

then the decision must say:

"The bail application was allowed."

If it says:

"the applicant be released on bail"

then report:

"The applicant was directed to be released on bail."

------------------------------------------------------------

3. DO NOT INFER A CRIMINAL SENTENCE.

If the judgment is a bail application, report the bail outcome.

Do NOT convert:

"Bail granted"

into:

"Accused convicted"

or:

"Punishment = imprisonment".

------------------------------------------------------------

4. PUNISHMENT

Only report punishment if the retrieved judgment explicitly
states a sentence or punishment.

Otherwise:

"Not stated"

------------------------------------------------------------

5. OUTCOME

Use the explicit outcome from the court order.

Examples:

"Bail granted"
"Bail denied"
"Anticipatory bail granted"
"Anticipatory bail denied"
"Convicted"
"Acquitted"
"Appeal allowed"
"Appeal dismissed"
"Petition allowed"
"Petition dismissed"
"Application disposed"
"Not stated"

Do not invent an outcome.

------------------------------------------------------------

6. DECISION NOT AVAILABLE

You may use:

"Decision not available in the retrieved judgment text."

ONLY when:

- there is no explicit court order,
- and the retrieved text does not contain a reliable final decision.

If an explicit court order is present, NEVER use that phrase.

------------------------------------------------------------

7. RELEVANT ISSUE

Describe only the issue actually supported by the retrieved
judgment.

Do not invent facts.

------------------------------------------------------------

8. WHY RELEVANT

Explain why the previous judgment is relevant to the current
case based only on the supplied evidence.

------------------------------------------------------------

9. RELEVANCE SCORE

Use the FAISS similarity value supplied in the evidence.

Do not invent another numeric score.

------------------------------------------------------------

10. LEGAL CAUTION

Do not state that the previous judgment determines the result
of the current case.

It is only a previous judgment with potentially similar facts
or legal issues.


============================================================
OUTPUT FORMAT
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

Do not add markdown.
Do not add ```json.
Do not add explanations outside JSON.
"""


# ============================================================
# EXTRACT JSON FROM GROQ RESPONSE
# ============================================================

def parse_json_response(response_text):

    response_text = response_text.strip()

    # Remove accidental markdown fences
    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:
        return json.loads(response_text)

    except json.JSONDecodeError:

        # Try extracting the JSON object
        start = response_text.find("{")
        end = response_text.rfind("}")

        if start != -1 and end != -1:

            json_text = response_text[start:end + 1]

            try:
                return json.loads(json_text)

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Groq returned invalid JSON:\n\n"
            + response_text
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CASETAG - PREVIOUS JUDGMENT + GROQ ANALYSIS")
    print("=" * 70)

    print("\nCurrent Case:")
    print(CASE_QUERY)

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEARCHING PREVIOUS JUDGMENTS")
    print("=" * 70)

    judgments = retrieve_previous_judgments(
        CASE_QUERY,
        k=5
    )

    print(f"\nRetrieved {len(judgments)} previous judgments.")

    if not judgments:
        print("No previous judgments found.")
        return

    # --------------------------------------------------------
    # SHOW RETRIEVAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVED JUDGMENTS")
    print("=" * 70)

    for i, judgment in enumerate(judgments, start=1):

        print(f"\n[{i}]")

        print(
            "Case:",
            judgment.get("case_title")
        )

        print(
            "Court:",
            judgment.get("court_name")
        )

        print(
            "Date:",
            judgment.get("decision_date")
        )

        print(
            "Docket:",
            judgment.get("docket_number")
        )

        print(
            "Similarity:",
            judgment.get("similarity")
        )

        print(
            "Matched chunk:",
            judgment.get("matched_chunk_id")
        )

        order_chunks = judgment.get("order_chunks", [])

        print(
            "Order chunks:",
            [
                chunk.get("chunk_id")
                if isinstance(chunk, dict)
                else "text"
                for chunk in order_chunks
            ]
        )

        # Show actual court order
        if order_chunks:

            print("\nPossible Court Order:\n")

            for order_chunk in order_chunks:

                if isinstance(order_chunk, dict):

                    text = (
                        order_chunk.get("text")
                        or order_chunk.get("chunk_text")
                        or order_chunk.get("content")
                        or ""
                    )

                else:
                    text = str(order_chunk)

                print(
                    truncate_text(text, 2500)
                )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    judgment_context = build_judgment_context(
        judgments
    )

    print("\n" + "=" * 70)
    print("GROQ CONTEXT SIZE")
    print("=" * 70)

    print(
        "Characters:",
        len(judgment_context)
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = create_prompt(
        CASE_QUERY,
        judgment_context
    )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SENDING JUDGMENTS TO GROQ")
    print("=" * 70)

    try:

        response = llm.invoke(prompt)

        raw_output = response.content

    except Exception as e:

        print("\nGROQ ERROR:")
        print(e)

        return

    # --------------------------------------------------------
    # RAW OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW GROQ OUTPUT")
    print("=" * 70)

    print(raw_output)

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    try:

        structured_output = parse_json_response(
            raw_output
        )

    except Exception as e:

        print("\nJSON PARSING ERROR:")
        print(e)

        return

    # --------------------------------------------------------
    # FINAL STRUCTURED OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRUCTURED PREVIOUS JUDGMENTS")
    print("=" * 70)

    print(
        json.dumps(
            structured_output,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # HUMAN-READABLE SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CASETAG - PREVIOUS JUDGMENT SUMMARY")
    print("=" * 70)

    previous_judgments = structured_output.get(
        "previous_judgments",
        []
    )

    for i, judgment in enumerate(
        previous_judgments,
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"PREVIOUS JUDGMENT {i}")
        print("-" * 70)

        print(
            "\nCase:",
            judgment.get("case_title", "Not stated")
        )

        print(
            "Court:",
            judgment.get("court", "Not stated")
        )

        print(
            "Decision Date:",
            judgment.get("decision_date", "Not stated")
        )

        print(
            "Docket Number:",
            judgment.get("docket_number", "Not stated")
        )

        print(
            "Relevance:",
            judgment.get("relevance", "Not stated")
        )

        print(
            "\nRelevant Issue:",
            judgment.get("relevant_issue", "Not stated")
        )

        print(
            "\nCourt's Decision:",
            judgment.get("court_decision", "Not stated")
        )

        print(
            "\nOutcome:",
            judgment.get("outcome", "Not stated")
        )

        print(
            "\nPunishment:",
            judgment.get("punishment", "Not stated")
        )

        print(
            "\nWhy Relevant:",
            judgment.get("why_relevant", "Not stated")
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()