
# ============================================================
# CASETag LEGAL AI - STREAMLIT FRONTEND
# ============================================================

import streamlit as st
import fitz

# ============================================================
# IMPORT RAG FUNCTIONS FROM rag.py
# ============================================================

from rag import retrieve, build_context, generate_answer
import rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASETag Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# NO BACKGROUND IMAGE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        font-weight: 600;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    textarea {
        border-radius: 12px !important;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stRadio"] {
        padding: 12px;
        border-radius: 12px;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ CASETag Legal AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Legal Issue and Section Retrieval Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ CASETag System")

    st.success(
        "FAISS Vector Database Loaded"
    )

    st.write(
        f"📚 Knowledge Vectors: "
        f"**{rag.index.ntotal:,}**"
    )

    st.write(
        f"🤖 LLM: "
        f"**{rag.LLM_MODEL}**"
    )

    st.write(
        f"🔤 Embeddings: "
        f"**{rag.EMBED_MODEL}**"
    )

    st.write(
        "🔎 RAG Framework: **LangChain**"
    )

    st.write(
        "🗂️ Vector Store: **FAISS**"
    )

    st.divider()

    st.subheader("🔄 RAG Pipeline")

    st.write("1️⃣ User Input")
    st.write("2️⃣ LangChain")
    st.write("3️⃣ Ollama Embeddings")
    st.write("4️⃣ FAISS Retrieval")
    st.write("5️⃣ Relevant Legal Provisions")
    st.write("6️⃣ Llama 3.2")
    st.write("7️⃣ Legal Analysis")

    st.divider()

    st.info(
        "CASETag retrieves relevant Indian legal provisions "
        "using LangChain and FAISS before generating an "
        "AI-based response with Llama 3.2."
    )


# ============================================================
# INPUT METHOD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Choose Input Method'
    '</div>',
    unsafe_allow_html=True
)

input_mode = st.radio(
    "Input Type",
    [
        "💬 Enter Legal Query",
        "📄 Upload Case PDF"
    ],
    horizontal=True
)


# ============================================================
# QUERY MODE
# ============================================================

if input_mode == "💬 Enter Legal Query":

    st.subheader(
        "💬 Ask a Legal Question"
    )

    query_text = st.text_area(
        "Enter your legal query",

        placeholder=(
            "Example:\n"
            "What is the punishment for theft?\n\n"
            "What punishment is provided for voluntarily "
            "causing hurt?"
        ),

        height=150
    )

    st.caption(
        "You can ask your question using natural language."
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        analyze_button = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )


    # ========================================================
    # CLEAR
    # ========================================================

    if clear_button:

        st.rerun()


    # ========================================================
    # ANALYZE QUERY
    # ========================================================

    if analyze_button:

        if not query_text.strip():

            st.warning(
                "Please enter a legal query first."
            )

            st.stop()


        user_query = query_text.strip()


        # ====================================================
        # FAISS RETRIEVAL
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 FAISS Retrieval"
        )

        with st.spinner(
            "Searching the legal knowledge base..."
        ):

            try:

                results = retrieve(
                    user_query
                )

            except Exception as e:

                st.error(
                    f"Retrieval error: {e}"
                )

                st.stop()


        if not results:

            st.warning(
                "No relevant legal provisions found."
            )

            st.stop()


        st.success(
            f"Retrieved {len(results)} "
            f"unique legal provisions."
        )


        # ====================================================
        # RETRIEVED LEGAL PROVISIONS
        # ====================================================

        st.subheader(
            "📚 Retrieved Legal Provisions"
        )


        for result in results:

            item = result["metadata"]


            act_name = item.get(
                "act_name",
                item.get(
                    "act",
                    "Unknown Act"
                )
            )


            section = item.get(
                "section",
                "Unknown"
            )


            heading = item.get(
                "heading",
                ""
            )


            text = item.get(
                "text",
                ""
            )


            source = item.get(
                "source_url",
                ""
            )


            with st.expander(
                f"Rank {result['rank']} | "
                f"{act_name} | "
                f"Section {section}"
            ):

                st.write(
                    f"**Heading:** {heading}"
                )


                st.write(
                    f"**FAISS Similarity Score:** "
                    f"{result['score']:.4f}"
                )


                st.write(
                    "**Legal Provision:**"
                )


                st.write(text)


                if source:

                    st.markdown(
                        f"**Source:** "
                        f"[{source}]({source})"
                    )


        # ====================================================
        # LLAMA LEGAL ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI Legal Analysis"
        )


        context = build_context(
            results
        )


        with st.spinner(
            "Llama 3.2 is generating the legal answer..."
        ):

            try:

                answer = generate_answer(
                    user_query,
                    context
                )

            except Exception as e:

                st.error(
                    f"LLM generation error: {e}"
                )

                st.stop()


        st.success(
            "Legal answer generated successfully."
        )


        st.markdown(answer)


# ============================================================
# PDF MODE
# ============================================================

else:

    st.subheader(
        "📄 Upload Case Filing"
    )


    uploaded_file = st.file_uploader(
        "Upload a legal case PDF",
        type=["pdf"]
    )


    pdf_text = ""


    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


        # ====================================================
        # EXTRACT PDF TEXT
        # ====================================================

        with st.spinner(
            "Extracting text from case PDF..."
        ):

            try:

                pdf_bytes = uploaded_file.getvalue()


                document = fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                )


                pages = []


                for page in document:

                    text = page.get_text()


                    if text:

                        pages.append(text)


                document.close()


                pdf_text = "\n".join(pages)


            except Exception as e:

                st.error(
                    f"PDF extraction error: {e}"
                )

                pdf_text = ""


        # ====================================================
        # DISPLAY EXTRACTED TEXT
        # ====================================================

        if pdf_text.strip():

            st.subheader(
                "📑 Extracted Case Text"
            )


            st.text_area(
                "Case content",
                pdf_text,
                height=250
            )

        else:

            st.warning(
                "No readable text was found in this PDF."
            )


    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        analyze_button = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True
        )


    with col2:

        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )


    # ========================================================
    # CLEAR
    # ========================================================

    if clear_button:

        st.rerun()


    # ========================================================
    # ANALYZE PDF
    # ========================================================

    if analyze_button:

        if uploaded_file is None:

            st.warning(
                "Please upload a case PDF first."
            )

            st.stop()


        if not pdf_text.strip():

            st.error(
                "The uploaded PDF contains no readable text."
            )

            st.stop()


        # ====================================================
        # CASE RETRIEVAL
        # ====================================================

        st.divider()

        st.subheader(
            "🔎 FAISS Case Retrieval"
        )


        # Use first 3000 characters as retrieval query
        case_query = pdf_text[:3000]


        with st.spinner(
            "Searching the legal knowledge base..."
        ):

            try:

                results = retrieve(
                    case_query
                )

            except Exception as e:

                st.error(
                    f"Retrieval error: {e}"
                )

                st.stop()


        if not results:

            st.warning(
                "No relevant legal provisions found."
            )

            st.stop()


        st.success(
            f"Retrieved {len(results)} "
            f"unique legal provisions."
        )


        # ====================================================
        # DISPLAY RETRIEVED PROVISIONS
        # ====================================================

        st.subheader(
            "📚 Retrieved Legal Provisions"
        )


        for result in results:

            item = result["metadata"]


            act_name = item.get(
                "act_name",
                item.get(
                    "act",
                    "Unknown Act"
                )
            )


            section = item.get(
                "section",
                "Unknown"
            )


            heading = item.get(
                "heading",
                ""
            )


            text = item.get(
                "text",
                ""
            )


            source = item.get(
                "source_url",
                ""
            )


            with st.expander(
                f"Rank {result['rank']} | "
                f"{act_name} | "
                f"Section {section}"
            ):

                st.write(
                    f"**Heading:** {heading}"
                )


                st.write(
                    f"**FAISS Similarity Score:** "
                    f"{result['score']:.4f}"
                )


                st.write(
                    "**Legal Provision:**"
                )


                st.write(text)


                if source:

                    st.markdown(
                        f"**Source:** "
                        f"[{source}]({source})"
                    )


        # ====================================================
        # LLAMA CASE ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI Case Analysis"
        )


        context = build_context(
            results
        )


        with st.spinner(
            "Llama 3.2 is analyzing the case..."
        ):

            try:

                answer = generate_answer(
                    pdf_text,
                    context
                )

            except Exception as e:

                st.error(
                    f"LLM generation error: {e}"
                )

                st.stop()


        st.success(
            "Case analysis generated successfully."
        )


        st.markdown(answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ CASETag Legal AI is an educational/research prototype. "
    "AI-generated responses should not be treated as professional "
    "legal advice."
)
