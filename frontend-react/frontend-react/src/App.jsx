
import { useRef, useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  NavLink,
  useLocation,
} from "react-router-dom";

import "./App.css";
import legalAiImage from "./assets/images.jpg";

const API_URL = "http://127.0.0.1:8000";

/* =========================================================
   LOGO
========================================================= */

function Logo() {
  return (
    <Link to="/" className="brand">
      <div className="brand-scale">
        <svg viewBox="0 0 64 64" aria-hidden="true">
          <path d="M32 8v40" />
          <path d="M20 48h24" />
          <path d="M25 53h14" />
          <path d="M14 17h36" />
          <path d="M32 8l-5 5h10z" />
          <path d="M14 17L6 31h16z" />
          <path d="M50 17l-8 14h16z" />
          <path d="M6 31c0 5 16 5 16 0" />
          <path d="M42 31c0 5 16 5 16 0" />
          <circle cx="32" cy="7" r="2" />
        </svg>
      </div>

      <div className="brand-text">
        <div className="brand-name">
          CASE<span>Tag</span>
        </div>
        <div className="brand-subtitle">LEGAL AI</div>
      </div>
    </Link>
  );
}

/* =========================================================
   NAVBAR
========================================================= */

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <Logo />

      <div className="nav-links">
        <NavLink
          to="/"
          className={({ isActive }) =>
            isActive && location.pathname === "/"
              ? "nav-link active"
              : "nav-link"
          }
        >
          Home
        </NavLink>

        <NavLink
          to="/legal-query"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          Legal Query
        </NavLink>

        <NavLink
          to="/case-analysis"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          Case Analysis
        </NavLink>

        <NavLink
          to="/about"
          className={({ isActive }) =>
            isActive ? "nav-link active" : "nav-link"
          }
        >
          About
        </NavLink>
      </div>

      <div className="nav-badge">
        <span>✦</span>
        AI POWERED
      </div>
    </nav>
  );
}

/* =========================================================
   FOOTER
========================================================= */

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-brand">
        <Logo />
      </div>

      <div className="footer-tagline">
        Legal AI for a more informed India.
      </div>

      <div className="footer-links">
        <Link to="/">Home</Link>
        <Link to="/legal-query">Legal Query</Link>
        <Link to="/case-analysis">Case Analysis</Link>
        <Link to="/about">About</Link>
      </div>

      <div className="footer-copy">
        © 2026 CASETag. All rights reserved.
        <br />
        For educational and research purposes only.
      </div>
    </footer>
  );
}

/* =========================================================
   LEGAL VISUAL
========================================================= */

function LegalVisual() {
  return (
    <div className="legal-visual legal-image-visual">
      <div className="visual-glow glow-a"></div>
      <div className="visual-glow glow-b"></div>

      <div className="legal-image-frame">
        <img
          src={legalAiImage}
          alt="AI Powered Legal Technology"
          className="legal-ai-hero-image"
        />

        <div className="legal-image-overlay">
          <span>AI · LAW · INDIA</span>
          <strong>Legal Intelligence</strong>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   ICONS
========================================================= */

function BookIcon() {
  return (
    <svg viewBox="0 0 48 48">
      <path d="M8 9h12c5 0 8 3 8 8v23c-2-3-5-4-8-4H8z" />
      <path d="M40 9H28c-5 0-8 3-8 8v23c2-3 5-4 8-4h12z" />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg viewBox="0 0 48 48">
      <path d="M24 5l3.5 12.5L40 21l-12.5 3.5L24 37l-3.5-12.5L8 21l12.5-3.5z" />
      <path d="M39 32l1.5 5.5L46 39l-5.5 1.5L39 46l-1.5-5.5L32 39l5.5-1.5z" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 48 48">
      <circle cx="21" cy="21" r="13" />
      <path d="M31 31l11 11" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 48 48">
      <path d="M24 5l16 6v11c0 10-6.5 17-16 21C14.5 39 8 32 8 22V11z" />
      <path d="M16 24l5 5 11-12" />
    </svg>
  );
}

/* =========================================================
   HOME
========================================================= */

function Home() {
  return (
    <>
      <Navbar />

      <main className="home-page">
        <section className="hero">
          <div className="hero-bg"></div>

          <div className="hero-content">
            <div className="eyebrow">
              LAW <span>•</span> TECHNOLOGY <span>•</span>
              A FAIRER TOMORROW
            </div>

            <h1>
              Your Legal
              <br />
              <span>Questions.</span>
              <br />
              Clearer Answers.
            </h1>

            <p className="hero-description">
              AI-assisted research for Indian laws,
              sections and case documents.
            </p>

            <div className="home-actions">
              <Link to="/legal-query" className="action-card">
                <div className="action-card-top">
                  <div className="round-icon">
                    <svg viewBox="0 0 48 48">
                      <path d="M10 12h28v21H22l-9 8v-8h-3z" />
                      <circle cx="18" cy="22" r="1.5" />
                      <circle cx="24" cy="22" r="1.5" />
                      <circle cx="30" cy="22" r="1.5" />
                    </svg>
                  </div>

                  <div className="arrow-circle">↗</div>
                </div>

                <h2>Ask a Legal Question</h2>

                <p>
                  Get relevant Indian Acts,
                  sections and legal information.
                </p>

                <div className="action-button">
                  Start a Query
                  <span>→</span>
                </div>

                <small>ASK · SEARCH · UNDERSTAND</small>
              </Link>

              <Link
                to="/case-analysis"
                className="action-card action-card-two"
              >
                <div className="action-card-top">
                  <div className="round-icon">
                    <svg viewBox="0 0 48 48">
                      <path d="M13 6h17l8 8v28H13z" />
                      <path d="M30 6v9h8" />
                      <path d="M19 24h12" />
                      <path d="M19 30h12" />
                    </svg>
                  </div>

                  <div className="arrow-circle">↗</div>
                </div>

                <h2>Analyze a Case Document</h2>

                <p>
                  Upload a case document and discover
                  relevant legal provisions and
                  previous judgments.
                </p>

                <div className="action-button">
                  Upload &amp; Analyze
                  <span>→</span>
                </div>

                <small>UPLOAD · ANALYZE · INSIGHTS</small>
              </Link>
            </div>
          </div>

          <LegalVisual />
        </section>

        <section className="feature-strip">
          <div className="feature-item">
            <div className="feature-icon">
              <BookIcon />
            </div>

            <div>
              <h3>50K+</h3>
              <strong>Legal Provisions</strong>
              <p>Indian Acts &amp; Sections</p>
            </div>
          </div>

          <div className="feature-divider"></div>

          <div className="feature-item">
            <div className="feature-icon">
              <SparkIcon />
            </div>

            <div>
              <h3>AI-Powered</h3>
              <strong>Legal Analysis</strong>
              <p>Context-aware insights</p>
            </div>
          </div>

          <div className="feature-divider"></div>

          <div className="feature-item">
            <div className="feature-icon">
              <SearchIcon />
            </div>

            <div>
              <h3>Relevant</h3>
              <strong>Sources</strong>
              <p>Retrieved legal references</p>
            </div>
          </div>

          <div className="feature-divider"></div>

          <div className="feature-item">
            <div className="feature-icon">
              <ShieldIcon />
            </div>

            <div>
              <h3>Secure</h3>
              <strong>Your Data</strong>
              <p>Built with privacy in mind</p>
            </div>
          </div>
        </section>

        <section className="home-info">
          <div>
            <span className="section-label">
              CASETAG LEGAL AI
            </span>

            <h2>
              Legal knowledge,
              <br />
              made more accessible.
            </h2>
          </div>

          <p>
            CASETag combines semantic retrieval
            with AI-generated explanations to
            help users discover relevant Indian
            legal provisions from natural-language
            questions and case documents.
          </p>
        </section>
      </main>

      <Footer />
    </>
  );
}

/* =========================================================
   BACK HOME
========================================================= */

function BackHome() {
  return (
    <Link to="/" className="back-home">
      ← Back to Home
    </Link>
  );
}

/* =========================================================
   ANSWER RENDERER
========================================================= */

function renderAnswerContent(answer) {
  if (!answer) {
    return (
      <p className="empty-answer">
        No answer was generated.
      </p>
    );
  }

  const labels = [
    "LEGAL ISSUE:",
    "APPLICABLE ACT:",
    "RELEVANT SECTION:",
    "OFFENCE STATUS:",
    "PUNISHMENT:",
    "EXPLANATION:",
    "OTHER RELEVANT PROVISIONS:",
  ];

  return answer.split("\n").map((line, index) => {
    const text = line.trim();

    if (!text) {
      return (
        <div
          key={index}
          className="answer-gap"
        ></div>
      );
    }

    if (text.startsWith("SOURCES:")) {
      const sourceText = text
        .substring("SOURCES:".length)
        .trim();

      if (sourceText) {
        return (
          <div key={index} className="answer-source">
            <strong>SOURCES:</strong>

            <a
              href={sourceText}
              target="_blank"
              rel="noopener noreferrer"
              className="source-link"
            >
              View Legal Source →
            </a>
          </div>
        );
      }

      return (
        <div key={index} className="answer-source">
          <strong>SOURCES:</strong>
        </div>
      );
    }

    if (
      text.startsWith("http://") ||
      text.startsWith("https://")
    ) {
      return (
        <div key={index} className="answer-source">
          <a
            href={text}
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            View Legal Source →
          </a>
        </div>
      );
    }

    const label = labels.find((item) =>
      text.startsWith(item)
    );

    if (label) {
      return (
        <p key={index} className="answer-line">
          <strong>{label}</strong>
          {text.substring(label.length)}
        </p>
      );
    }

    return (
      <p key={index} className="answer-line">
        {text}
      </p>
    );
  });
}

/* =========================================================
   PROVISIONS
========================================================= */

function Provisions({ provisions }) {
  if (
    !Array.isArray(provisions) ||
    provisions.length === 0
  ) {
    return null;
  }

  return (
    <div className="provisions-section">
      <div className="provisions-heading">
        <span>RETRIEVED LAW</span>
        <h3>Relevant Legal Provisions</h3>
      </div>

      <div className="provisions-list">
        {provisions.map((provision, index) => (
          <div
            className="provision-card"
            key={index}
          >
            <div className="provision-index">
              {String(index + 1).padStart(2, "0")}
            </div>

            <div className="provision-body">
              <div className="provision-act">
                {provision.act_name ||
                  provision.act ||
                  "Indian Legal Provision"}
              </div>

              <div className="provision-section-title">
                Section {provision.section}

                {provision.heading &&
                  ` — ${provision.heading}`}
              </div>

              <p>{provision.text}</p>

              {provision.source_url && (
                <a
                  href={provision.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="source-link"
                >
                  View Legal Source →
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   STRUCTURED LEGAL ANALYSIS
========================================================= */

function StructuredLegalAnalysis({ analysis }) {
  if (!analysis) {
    return null;
  }

  const laws = Array.isArray(
    analysis.applicable_laws
  )
    ? analysis.applicable_laws
    : [];

  const issues = Array.isArray(
    analysis.legal_issues
  )
    ? analysis.legal_issues
    : [];

  const facts = Array.isArray(
    analysis.key_facts
  )
    ? analysis.key_facts
    : [];

  return (
    <div className="structured-analysis">

      {/* CASE SUMMARY */}

      {analysis.case_summary && (
        <div className="analysis-card analysis-summary-card">
          <div className="analysis-card-label">
            CASE SUMMARY
          </div>

          <p>{analysis.case_summary}</p>
        </div>
      )}

      {/* LEGAL ISSUE */}

      {issues.length > 0 && (
        <div className="analysis-card">
          <div className="analysis-card-label">
            LEGAL ISSUE
          </div>

          <div className="analysis-issue-list">
            {issues.map((issue, index) => (
              <p key={index}>{issue}</p>
            ))}
          </div>
        </div>
      )}

      {/* KEY FACTS */}

      {facts.length > 0 && (
        <div className="analysis-card">
          <div className="analysis-card-label">
            KEY FACTS
          </div>

          <ul className="analysis-list">
            {facts.map((fact, index) => (
              <li key={index}>{fact}</li>
            ))}
          </ul>
        </div>
      )}

      {/* APPLICABLE LAW */}

      {laws.length > 0 && (
        <div className="analysis-laws">
          <div className="analysis-section-heading">
            <span>APPLICABLE LAW</span>
          </div>

          <div className="law-cards">
            {laws.map((law, index) => (
              <div
                className="law-card"
                key={index}
              >
                <div className="law-card-content">

                  <div className="law-field">
                    <span className="law-field-label">
                      ACT:
                    </span>

                    <div className="law-field-value">
                      {law.act ||
                        law.act_name ||
                        "Indian Legal Provision"}
                    </div>
                  </div>

                  <div className="law-field">
                    <span className="law-field-label">
                      SECTION:
                    </span>

                    <div className="law-field-value">
                      Section {law.section}

                      {law.heading
                        ? ` — ${law.heading}`
                        : ""}
                    </div>
                  </div>

                  {law.punishment && (
                    <div className="law-field punishment-field">
                      <span className="law-field-label">
                        PUNISHMENT:
                      </span>

                      <div className="law-field-value">
                        {law.punishment}
                      </div>
                    </div>
                  )}

                  {law.source_url && (
                    <div className="law-source">
                      <span className="law-field-label">
                        SOURCE:
                      </span>

                      <a
                        href={law.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        View Legal Source →
                      </a>
                    </div>
                  )}

                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

/* =========================================================
   PREVIOUS JUDGMENTS
========================================================= */

function PreviousJudgments({ judgments }) {
  const [openJudgment, setOpenJudgment] = useState(null);

  if (
    !Array.isArray(judgments) ||
    judgments.length === 0
  ) {
    return null;
  }

  const toggleJudgment = (index) => {
    setOpenJudgment(
      openJudgment === index ? null : index
    );
  };

  return (
    <div className="previous-judgments-section">

      <div className="previous-judgments-heading">
        <span>PREVIOUS JUDGMENT</span>

        <h3>Relevant Previous Judgments</h3>

        <p>
          Similar judgments retrieved from
          the CASETag judgment knowledge base.
        </p>
      </div>

      <div className="previous-judgments-list">

        {judgments.map((judgment, index) => {

          const isOpen = openJudgment === index;

          return (
            <div
              className="previous-judgment-card"
              key={index}
            >

              {/* ================================
                  JUDGMENT HEADER
              ================================= */}

              <div className="previous-judgment-header">

                <div className="previous-judgment-number">
                  {String(index + 1).padStart(2, "0")}
                </div>

                <div className="previous-judgment-title-wrap">

                  <span className="previous-judgment-label">
                    PREVIOUS JUDGMENT
                  </span>

                  <h4>
                    {judgment.case_title ||
                      "Case title unavailable"}
                  </h4>

                </div>

              </div>


              {/* ================================
                  BASIC INFORMATION
              ================================= */}

              <div className="previous-judgment-meta">

                <div className="judgment-meta-item">

                  <span>COURT</span>

                  <strong>
                    {judgment.court ||
                      judgment.court_name ||
                      "Not available"}
                  </strong>

                </div>


                <div className="judgment-meta-item">

                  <span>DECISION DATE</span>

                  <strong>
                    {judgment.decision_date ||
                      "Not available"}
                  </strong>

                </div>


                <div className="judgment-meta-item">

                  <span>DOCKET NUMBER</span>

                  <strong>
                    {judgment.docket_number ||
                      "Not available"}
                  </strong>

                </div>


                <div className="judgment-meta-item">

                  <span>RELEVANCE</span>

                  <strong>
                    {judgment.relevance ??
                      judgment.similarity ??
                      "Not available"}
                  </strong>

                </div>

              </div>


              {/* ================================
                  VIEW JUDGMENT BUTTON
              ================================= */}

              <button
                type="button"
                className="view-judgment-button"
                onClick={() =>
                  toggleJudgment(index)
                }
              >

                <span>
                  {isOpen
                    ? "Hide Judgment"
                    : "View Judgment"}
                </span>

                <span
                  className={
                    isOpen
                      ? "view-judgment-arrow rotate"
                      : "view-judgment-arrow"
                  }
                >
                  ↓
                </span>

              </button>


              {/* ================================
                  EXPANDED JUDGMENT DETAILS
              ================================= */}

              {isOpen && (

                <div className="judgment-expanded">

                  {/* RELEVANT ISSUE */}

                  {judgment.relevant_issue && (
                    <div className="judgment-detail">

                      <div className="judgment-detail-label">
                        RELEVANT ISSUE
                      </div>

                      <p>
                        {judgment.relevant_issue}
                      </p>

                    </div>
                  )}


                  {/* COURT DECISION */}

                  {judgment.court_decision && (
                    <div className="judgment-detail judgment-decision">

                      <div className="judgment-detail-label">
                        COURT'S DECISION
                      </div>

                      <p>
                        {judgment.court_decision}
                      </p>

                    </div>
                  )}


                  {/* OUTCOME + PUNISHMENT */}

                  <div className="judgment-outcome-grid">

                    {judgment.outcome && (
                      <div className="judgment-outcome-box">

                        <span>
                          OUTCOME
                        </span>

                        <strong>
                          {judgment.outcome}
                        </strong>

                      </div>
                    )}


                    {judgment.punishment && (
                      <div className="judgment-outcome-box">

                        <span>
                          PUNISHMENT
                        </span>

                        <strong>
                          {judgment.punishment}
                        </strong>

                      </div>
                    )}

                  </div>


                  {/* WHY RELEVANT */}

                  {judgment.why_relevant && (
                    <div className="judgment-detail">

                      <div className="judgment-detail-label">
                        WHY THIS JUDGMENT IS RELEVANT
                      </div>

                      <p>
                        {judgment.why_relevant}
                      </p>

                    </div>
                  )}


                  {/* ORIGINAL JUDGMENT */}

                  {(judgment.source_url ||
                    judgment.source_pdf_s3_url) && (

                    <div className="original-judgment">

                      <a
                        href={
                          judgment.source_url ||
                          judgment.source_pdf_s3_url
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="original-judgment-button"
                      >

                        Open Original Judgment
                        <span>↗</span>

                      </a>

                    </div>

                  )}

                </div>

              )}

            </div>
          );
        })}

      </div>
    </div>
  );
}

/* =========================================================
   LEGAL QUERY
========================================================= */

function LegalQuery() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/api/query`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            query: query.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Unable to process the query."
        );
      }

      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        answer: error.message,
        provisions: [],
        previous_judgments: [],
      });
    } finally {
      setLoading(false);
    }
  };

  const clearQuery = () => {
    setQuery("");
    setResult(null);
  };

  return (
    <>
      <Navbar />

      <main className="tool-page">
        <div className="tool-container">

          <BackHome />

          <div className="tool-heading">
            <span className="section-label">
              LEGAL QUERY
            </span>

            <h1>
              Ask. Search.
              <br />
              <span>Understand.</span>
            </h1>

            <p>
              Ask a question in natural
              language and discover relevant
              Indian legal provisions.
            </p>
          </div>

          <div className="query-layout">

            <div className="query-workspace">

              <div className="workspace-header">
                <div className="workspace-number">
                  01
                </div>

                <div>
                  <span>
                    YOUR LEGAL QUESTION
                  </span>

                  <h3>
                    What would you like to know?
                  </h3>
                </div>
              </div>

              <textarea
                value={query}
                onChange={(event) =>
                  setQuery(event.target.value)
                }
                placeholder="Example: What is the punishment for committing murder?"
              />

              <div className="query-actions">

                <button
                  className="primary-button"
                  onClick={handleQuery}
                  disabled={
                    loading ||
                    !query.trim()
                  }
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze Question →
                    </>
                  )}
                </button>

                <button
                  className="secondary-button"
                  onClick={clearQuery}
                >
                  Clear
                </button>

              </div>
            </div>

            <div className="tool-side-card">
              <div className="side-card-icon">
                ✦
              </div>

              <span>
                AI LEGAL ASSISTANT
              </span>

              <h3>
                From your question
                <br />
                to relevant law.
              </h3>

              <p>
                CASETag retrieves relevant legal
                provisions and uses AI to explain
                them in a clear, structured way.
              </p>

              <div className="side-card-line"></div>

              <small>
                RETRIEVAL · RELEVANCE · EXPLANATION
              </small>
            </div>

          </div>

          {/* =================================================
              LEGAL QUERY RESULT
          ================================================= */}

          {result && (
            <div className="result-container">

              <div className="result-header">
                <div>
                  <span>
                    AI LEGAL ANALYSIS
                  </span>

                  <h2>
                    Analysis Result
                  </h2>
                </div>

                <div className="result-status">
                  {result.success === false
                    ? "ERROR"
                    : "GENERATED"}
                </div>
              </div>

              {result.success === false ? (

                <div className="result-answer">
                  {renderAnswerContent(
                    result.answer
                  )}
                </div>

              ) : (

                <>

                  {/* STRUCTURED LEGAL ANALYSIS */}

                  {result.analysis ? (

                    <StructuredLegalAnalysis
                      analysis={result.analysis}
                    />

                  ) : (

                    <>
                      <div className="result-answer">
                        {renderAnswerContent(
                          result.answer
                        )}
                      </div>

                      <Provisions
                        provisions={result.provisions}
                      />
                    </>

                  )}

                  {/* PREVIOUS JUDGMENTS */}

                  <PreviousJudgments
                    judgments={
                      result.previous_judgments
                    }
                  />

                </>

              )}

            </div>
          )}

        </div>
      </main>

      <Footer />
    </>
  );
}

/* =========================================================
   CASE ANALYSIS
========================================================= */

function CaseAnalysis() {
  const [
    selectedFile,
    setSelectedFile,
  ] = useState(null);

  const [
    result,
    setResult,
  ] = useState(null);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const fileInputRef = useRef(null);

  /* -------------------------------------------------------
     FILE SELECTION
  ------------------------------------------------------- */

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    const allowedExtensions = [
      ".pdf",
      ".doc",
      ".docx",
      ".txt",
    ];

    const fileName = file.name.toLowerCase();

    const isAllowed = allowedExtensions.some(
      (extension) =>
        fileName.endsWith(extension)
    );

    if (!isAllowed) {
      alert(
        "Please upload a PDF, Word, or TXT file."
      );

      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setResult(null);
  };

  /* -------------------------------------------------------
     CLEAR FILE
  ------------------------------------------------------- */

  const clearFile = () => {
    setSelectedFile(null);
    setResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  /* -------------------------------------------------------
     ANALYZE CASE
  ------------------------------------------------------- */

  const analyzeCase = async () => {
    if (!selectedFile) {
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        selectedFile
      );

      const response = await fetch(
        `${API_URL}/api/analyze-case`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Unable to analyze the case."
        );
      }

      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        answer: error.message,
        provisions: [],
        previous_judgments: [],
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />

      <main className="tool-page case-tool-page">
        <div className="tool-container">

          <BackHome />

          <div className="tool-heading">
            <span className="section-label">
              CASE ANALYSIS
            </span>

            <h1>
              Turn Case Documents
              <br />
              <span>
                Into Legal Insights.
              </span>
            </h1>

            <p>
              Upload your case document and
              identify relevant legal provisions
              and previous judgments using AI.
            </p>
          </div>

          <div className="case-layout">

            {/* UPLOAD WORKSPACE */}

            <div className="upload-workspace">

              <div className="upload-icon-large">
                <svg viewBox="0 0 48 48">
                  <path d="M13 5h17l8 8v30H13z" />
                  <path d="M30 5v9h8" />
                  <path d="M19 24h12" />
                  <path d="M19 30h12" />
                </svg>
              </div>

              <span className="upload-label">
                CASE DOCUMENT
              </span>

              <h2>
                Upload your case document
              </h2>

              <p>
                Upload a PDF, Word document or
                TXT file containing the case
                facts, judgment or legal document
                you want CASETag to analyze.
              </p>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.doc,.docx,.txt,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
                onChange={handleFileChange}
                hidden
              />

              {!selectedFile ? (

                <button
                  className="choose-file-button"
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  Choose File
                  <span>→</span>
                </button>

              ) : (

                <div className="selected-file">

                  <div className="selected-file-icon">
                    {selectedFile.name
                      .split(".")
                      .pop()
                      .toUpperCase()}
                  </div>

                  <div className="selected-file-info">
                    <strong>
                      {selectedFile.name}
                    </strong>

                    <small>
                      {(
                        selectedFile.size /
                        1024 /
                        1024
                      ).toFixed(2)}{" "}
                      MB
                    </small>
                  </div>

                  <button
                    className="remove-file"
                    onClick={clearFile}
                    aria-label="Remove selected file"
                  >
                    ×
                  </button>

                </div>

              )}

              {selectedFile && (
                <button
                  className="analyze-case-button"
                  onClick={analyzeCase}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing Case...
                    </>
                  ) : (
                    <>
                      Analyze Case →
                    </>
                  )}
                </button>
              )}

              <div className="upload-note">
                <span>✓</span>

                Supported formats:
                PDF · DOC · DOCX · TXT
              </div>

            </div>

            {/* SIDE PANEL */}

            <div className="case-side-panel">

              <div className="case-side-top">
                <span>
                  HOW IT WORKS
                </span>

                <div>
                  03 STEPS
                </div>
              </div>

              <div className="step">
                <strong>01</strong>

                <div>
                  <h3>Upload</h3>

                  <p>
                    Select your case document.
                  </p>
                </div>
              </div>

              <div className="step">
                <strong>02</strong>

                <div>
                  <h3>Retrieve</h3>

                  <p>
                    Find relevant legal provisions
                    and previous judgments.
                  </p>
                </div>
              </div>

              <div className="step">
                <strong>03</strong>

                <div>
                  <h3>Understand</h3>

                  <p>
                    Get an AI-assisted legal analysis
                    grounded in retrieved evidence.
                  </p>
                </div>
              </div>

            </div>

          </div>

          {/* =================================================
              CASE ANALYSIS RESULT
          ================================================= */}

          {result && (
            <div className="result-container case-result-container">

              <div className="result-header">
                <div>
                  <span>
                    CASE ANALYSIS RESULT
                  </span>

                  <h2>
                    Legal Insights
                  </h2>
                </div>

                <div className="result-status">
                  {result.success === false
                    ? "ERROR"
                    : "GENERATED"}
                </div>
              </div>

              {result.success === false ? (

                <div className="result-answer">
                  <p className="answer-line">
                    {result.answer ||
                      "Unable to analyze the case."}
                  </p>
                </div>

              ) : (

                <>

                  {/* STRUCTURED CASE ANALYSIS */}

                  {result.analysis ? (

                    <StructuredLegalAnalysis
                      analysis={result.analysis}
                    />

                  ) : (

                    <>
                      <div className="result-answer">
                        {renderAnswerContent(
                          result.answer
                        )}
                      </div>

                      <Provisions
                        provisions={
                          result.provisions
                        }
                      />
                    </>

                  )}

                  {/* PREVIOUS JUDGMENTS */}

                  <PreviousJudgments
                    judgments={
                      result.previous_judgments
                    }
                  />

                </>

              )}

            </div>
          )}

        </div>
      </main>

      <Footer />
    </>
  );
}

/* =========================================================
   ABOUT
========================================================= */

function About() {
  return (
    <>
      <Navbar />

      <main className="about-page">
        <div className="about-container">

          <BackHome />

          <div className="about-hero">
            <div>
              <span className="section-label">
                ABOUT CASETAG
              </span>

              <h1>
                Legal knowledge,
                <br />
                <span>
                  made more accessible.
                </span>
              </h1>
            </div>

            <p>
              CASETag is an AI-powered legal
              research assistant designed to help
              users discover relevant Indian Acts,
              sections and legal provisions using
              natural-language queries and case
              documents.
            </p>
          </div>

          <div className="about-grid">

            <div className="about-card">
              <span>01</span>

              <h2>Retrieval</h2>

              <p>
                CASETag uses semantic search to
                retrieve relevant Indian legal
                provisions from its legal
                knowledge base.
              </p>
            </div>

            <div className="about-card">
              <span>02</span>

              <h2>Analysis</h2>

              <p>
                The retrieved legal context is
                provided to an AI model for
                structured legal explanation.
              </p>
            </div>

            <div className="about-card">
              <span>03</span>

              <h2>Previous Judgments</h2>

              <p>
                CASETag retrieves similar previous
                judgments and presents the court,
                decision and outcome when supported
                by the retrieved judgment evidence.
              </p>
            </div>

          </div>

          <div className="about-bottom">
            <div>
              <span className="section-label">
                TECHNOLOGY
              </span>

              <h2>
                Retrieval-Augmented
                <br />
                Generation.
              </h2>
            </div>

            <p>
              CASETag combines vector-based
              retrieval with large language
              models to generate context-aware
              explanations while grounding
              responses in retrieved legal
              provisions and previous judgments.
            </p>
          </div>

        </div>
      </main>

      <Footer />
    </>
  );
}

/* =========================================================
   APP
========================================================= */

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/legal-query"
          element={<LegalQuery />}
        />

        <Route
          path="/case-analysis"
          element={<CaseAnalysis />}
        />

        <Route
          path="/about"
          element={<About />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;