# DocuAudit AI: Procurement & Compliance Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203-orange.svg)

**DocuAudit AI** is an enterprise-grade automated auditing tool designed for legal, procurement, and compliance teams. It utilizes a sophisticated two-stage Retrieval-Augmented Generation (RAG) pipeline to instantly scan Master Service Agreements (MSAs), Non-Disclosure Agreements (NDAs), and vendor contracts for hidden risks, financial liabilities, and unreciprocal terms.

## Core Features

* **Zero-Trust Verifiable Citations:** DocuAudit doesn't just give answers; it proves them. Every claim made by the AI is backed by a cryptographic ID linked to the exact structural section and page number of the source document.
* **Two-Stage Hybrid Retrieval:** Combines sparse (BM25) and dense (Chroma DB) vector retrieval to ensure both keyword exact-matching and semantic understanding.
* **Semantic Reranking:** Integrates FlashRank to compress and re-order retrieved clauses, ensuring only the most highly relevant context is passed to the LLM.
* **Structured Output Engine:** Enforces strict JSON schemas on the LLM (Llama-3.3-70b-versatile via Groq) to prevent hallucinations and ensure standardized, color-coded risk reporting.
* **Enterprise Data Isolation:** Features a secure cloud gateway that completely isolates user sessions, purging vector databases and memory caches instantly upon workspace resets.

## Architecture

1.  **Ingestion & Parsing:** PDFs are structurally parsed using Regex boundaries to identify distinct legal clauses (e.g., "Section 1", "Article IV").
2.  **Embedding:** Text is embedded locally using high-performance lightweight models (`all-MiniLM-L6-v2`).
3.  **Hybrid RAG:** The Ensemble Retriever fetches candidates, which are then passed through the FlashRank cross-encoder.
4.  **Inference:** Context is packaged with strict instructions and sent to the Groq API for near-instantaneous JSON compliance parsing.

## Getting Started (Local Development)

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/docuaudit-ai.git](https://github.com/your-username/docuaudit-ai.git)
cd docuaudit-ai
