# ============================================================
# CASETag LEGAL AI
# FASTAPI BACKEND
# ============================================================

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from dotenv import load_dotenv

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import os
import tempfile


# ============================================================
# ACTS / SECTIONS RAG IMPORTS
# ============================================================

from rag import (
    retrieve,
    build_context,
    generate_structured_analysis,
    answer_query
)

import rag

# ============================================================
# PREVIOUS JUDGMENT RAG IMPORTS
# ============================================================

from retrieve_previous_judgments import (
    retrieve_previous_judgments,
    generate_previous_judgment_analysis
)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

from pdf_processor import (
    extract_text_from_document,
    clean_text
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CASETag Legal AI",
    description=(
        "AI-Powered Legal Issue and "
        "Section Retrieval Assistant"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class LegalQueryRequest(BaseModel):

    query: str


# ============================================================
# RAG INFORMATION
# ============================================================

def get_rag_info():

    try:

        import rag

        index = getattr(
            rag,
            "faiss_index",
            None
        )

        metadata = getattr(
            rag,
            "metadata",
            None
        )

        embedding_model = getattr(
            rag,
            "EMBEDDING_MODEL",
            "Unknown"
        )

        llm_model = getattr(
            rag,
            "LLM_MODEL",
            "Unknown"
        )

        return {

            "vectors": (
                int(index.ntotal)
                if index is not None
                else 0
            ),

            "metadata": (
                len(metadata)
                if metadata is not None
                else 0
            ),

            "embedding_model": embedding_model,

            "llm_model": llm_model
        }


    except Exception as error:

        return {

            "vectors": 0,

            "metadata": 0,

            "embedding_model": "Unknown",

            "llm_model": "Unknown",

            "error": str(error)
        }


# ============================================================
# PROVISION NORMALIZATION
# ============================================================

def normalize_provision(item):

    if not isinstance(item, dict):

        return {

            "act_name": "",

            "act": "",

            "section": "",

            "heading": "",

            "text": str(item),

            "source_url": ""
        }


    return {

        "act_name": item.get(
            "act_name",
            item.get(
                "act",
                ""
            )
        ),

        "act": item.get(
            "act",
            ""
        ),

        "section": item.get(
            "section",
            ""
        ),

        "heading": item.get(
            "heading",
            ""
        ),

        "text": item.get(
            "text",
            item.get(
                "chunk",
                ""
            )
        ),

        "source_url": item.get(
            "source_url",
            item.get(
                "source",
                ""
            )
        )
    }


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "success": True,

        "message": "CASETag Legal AI API is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    info = get_rag_info()

    return {

        "success": True,

        "status": "healthy",

        **info
    }


# ============================================================
# LEGAL QUERY
# ============================================================

@app.post("/api/query")
def legal_query(
    request: LegalQueryRequest
):

    question = (
        request.query or ""
    ).strip()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Please enter a legal question."
        )


    try:

        # ====================================================
        # 1. ACTS / SECTIONS RETRIEVAL
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CASETAG - ACTS / SECTIONS RETRIEVAL"
        )

        print(
            "=" * 70
        )


        results = retrieve(
            question
        )


        # ====================================================
        # NO LEGAL PROVISIONS
        # ====================================================

        if not results:

            return {

                "success": True,

                "query": question,

                "answer": (
                    "No relevant legal provisions "
                    "were found in the CASETag "
                    "knowledge base."
                ),

                "analysis": None,

                "provisions": [],

                "previous_judgments": []
            }


        # ====================================================
        # 2. BUILD CONTEXT
        # ====================================================

        context = build_context(
            results
        )


        # ====================================================
        # 3. STRUCTURED LEGAL ANALYSIS
        # ====================================================

        analysis = (
            generate_structured_analysis(
                question,
                results
            )
        )


        answer = ""


        # ====================================================
        # 4. NORMALIZE PROVISIONS
        # ====================================================

        provisions = [

            normalize_provision(
                result
            )

            for result in results
        ]


        # ====================================================
        # 5. PREVIOUS JUDGMENT RETRIEVAL
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CASETAG - PREVIOUS JUDGMENT RETRIEVAL"
        )

        print(
            "=" * 70
        )


        previous_judgments_raw = (

            retrieve_previous_judgments(

                question,

                k=5
            )
        )


        print(

            "Retrieved "

            f"{len(previous_judgments_raw)} "

            "previous judgments."
        )


        # ====================================================
        # 6. PREVIOUS JUDGMENT ANALYSIS
        # ====================================================

        previous_judgments = []


        if previous_judgments_raw:

            previous_result = (

                generate_previous_judgment_analysis(

                    question,

                    previous_judgments_raw
                )
            )


            if isinstance(
                previous_result,
                dict
            ):

                previous_judgments = (

                    previous_result.get(

                        "previous_judgments",

                        []
                    )
                )


        # ====================================================
        # 7. FINAL RESPONSE
        # ====================================================

        return {

            "success": True,

            "query": question,

            "answer": answer,

            "analysis": analysis,

            "provisions": provisions,

            "previous_judgments":
                previous_judgments
        }


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except HTTPException:

        raise


    except ValueError as error:

        print(

            "LEGAL QUERY VALUE ERROR:",

            repr(error)
        )


        raise HTTPException(

            status_code=400,

            detail=str(error)
        )


    except Exception as error:

        print(

            "LEGAL QUERY ERROR:",

            repr(error)
        )


        raise HTTPException(

            status_code=500,

            detail=(

                "Unable to process the legal query: "

                f"{str(error)}"
            )
        )


# ============================================================
# CASE ANALYSIS
# PDF + DOC + DOCX + TXT
# ============================================================

@app.post("/api/analyze-case")
async def analyze_case(

    file: UploadFile = File(...)
):


    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail="No file was selected."
        )


    filename = file.filename.lower()


    # ========================================================
    # SUPPORTED FILE FORMATS
    # ========================================================

    allowed_extensions = [

        ".pdf",

        ".doc",

        ".docx",

        ".txt"
    ]


    file_extension = os.path.splitext(
        filename
    )[1]


    if file_extension not in allowed_extensions:

        raise HTTPException(

            status_code=400,

            detail=(

                "Unsupported file format. "

                "Please upload PDF, DOC, DOCX, or TXT."
            )
        )


    try:

        # ====================================================
        # READ UPLOADED FILE
        # ====================================================

        file_bytes = await file.read()


        if not file_bytes:

            raise HTTPException(

                status_code=400,

                detail="The uploaded file is empty."
            )


        # ====================================================
        # TEMPORARY FILE
        # ====================================================

        temp_file_path = None


        try:

            suffix = file_extension


            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=suffix

            ) as temp_file:

                temp_file.write(
                    file_bytes
                )


                temp_file_path = (
                    temp_file.name
                )


            # ------------------------------------------------
            # EXTRACT TEXT
            # ------------------------------------------------

            extracted_text = (

                extract_text_from_document(

                    temp_file_path
                )
            )


        finally:

            # ------------------------------------------------
            # DELETE TEMP FILE
            # ------------------------------------------------

            if (

                temp_file_path

                and os.path.exists(
                    temp_file_path
                )

            ):

                os.remove(
                    temp_file_path
                )


        # ====================================================
        # CHECK EXTRACTED TEXT
        # ====================================================

        if not extracted_text:

            raise HTTPException(

                status_code=400,

                detail=(

                    "No readable text could be extracted "

                    "from the uploaded document."
                )
            )


        # ====================================================
        # CLEAN TEXT
        # ====================================================

        case_text = clean_text(
            extracted_text
        )


        if not case_text.strip():

            raise HTTPException(

                status_code=400,

                detail=(

                    "The uploaded document does not contain "

                    "readable text."
                )
            )


        # ====================================================
        # LIMIT QUERY SIZE
        # ====================================================

        case_query = case_text[:6000]


        # ====================================================
        # 1. ACTS / SECTIONS RAG
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CASETAG - ACTS / SECTIONS RETRIEVAL"
        )

        print(
            "=" * 70
        )


        results = retrieve(
            case_query
        )


        analysis = None

        provisions = []


        if results:

            # ------------------------------------------------
            # GROQ LEGAL ANALYSIS
            # ------------------------------------------------

            analysis = (

                generate_structured_analysis(

                    case_text,

                    results
                )
            )


            # ------------------------------------------------
            # NORMALIZE PROVISIONS
            # ------------------------------------------------

            provisions = [

                normalize_provision(

                    result
                )

                for result in results
            ]


        else:

            print(
                "No relevant legal provisions found."
            )


        # ====================================================
        # 2. PREVIOUS JUDGMENTS RAG
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CASETAG - PREVIOUS JUDGMENT RETRIEVAL"
        )

        print(
            "=" * 70
        )


        previous_judgments_raw = (

            retrieve_previous_judgments(

                case_query,

                k=5
            )
        )


        print(

            "Retrieved "

            f"{len(previous_judgments_raw)} "

            "previous judgments."
        )


        # ====================================================
        # DEFAULT RESPONSE
        # ====================================================

        previous_judgments = []


        # ====================================================
        # 3. GROQ PREVIOUS JUDGMENT ANALYSIS
        # ====================================================

        if previous_judgments_raw:

            previous_result = (

                generate_previous_judgment_analysis(

                    case_query,

                    previous_judgments_raw
                )
            )


            if isinstance(

                previous_result,

                dict

            ):

                previous_judgments = (

                    previous_result.get(

                        "previous_judgments",

                        []
                    )
                )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {

            "success": True,

            "filename": file.filename,

            # Existing CaseTag analysis

            "answer": "",

            "analysis": analysis,

            "provisions": provisions,

            # Previous Judgment RAG

            "previous_judgments":
                previous_judgments
        }


    # ========================================================
    # ERRORS
    # ========================================================

    except HTTPException:

        raise


    except ValueError as error:

        print(

            "DOCUMENT / RAG ERROR:",

            repr(error)
        )


        raise HTTPException(

            status_code=400,

            detail=str(error)
        )


    except Exception as error:

        print(

            "CASE ANALYSIS ERROR:",

            repr(error)
        )


        raise HTTPException(

            status_code=500,

            detail=(

                "Unable to analyze the case: "

                f"{str(error)}"
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )