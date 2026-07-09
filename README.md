# DocuAudit AI: Automated Contract Analysis and Retrieval System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange.svg)

DocuAudit AI is a Retrieval-Augmented Generation (RAG) application designed to assist legal and procurement teams in reviewing structured documents such as vendor contracts, Master Service Agreements (MSAs), and Non-Disclosure Agreements (NDAs).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Challenges & Engineering Decisions](#challenges--engineering-decisions)
6. [Evaluation & Testing](#evaluation--testing)
7. [Limitations](#limitations)
8. [Future Improvements](#future-improvements)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Screenshots](#screenshots)
12. [Repository Structure](#repository-structure)

---

## Project Overview

Reviewing legal contracts is a highly manual, error-prone process. The difficulty stems from dense formatting, dispersed information (where a clause on page 2 interacts with a liability cap on page 10), and the sheer volume of text that must be parsed to identify specific financial or compliance risks. 

DocuAudit AI applies a multi-stage RAG architecture to address these challenges. Instead of relying entirely on an LLM's context window—which can lead to lost-in-the-middle phenomena and hallucinations—this system utilizes hybrid retrieval to extract the most relevant clauses, reranks them for precision, and forces the LLM to answer strictly based on the retrieved context with explicit source attribution.

## Key Features

* **PDF Contract Ingestion:** Parses standard text-based PDF documents into processable text data.
* **Clause-Aware Chunking:** Utilizes custom regular expressions to chunk documents by structural boundaries (e.g., "Section 1", "Article IV") rather than arbitrary character limits.
* **Hybrid Retrieval (BM25 + Chroma):** Combines sparse lexical search (BM25) for exact keyword matching with dense semantic search (Chroma vector database).
* **FlashRank Reranking:** Applies a cross-encoder to rerank retrieved chunks, improving the relevance of the context passed to the LLM.
* **Procurement and Compliance Analysis:** Pre-configured prompts specifically designed to identify financial liabilities, unilateral terms, and standard compliance risks.
* **Source Attribution:** Implements a deterministic hashing system to map LLM outputs back to specific document clauses and page numbers.
* **Audit Report Export:** Allows users to download the analysis and supporting citations as a structured text file.
* **Multi-Document Support:** Capable of indexing and retrieving across multiple uploaded contracts simultaneously.
* **Streamlit Interface:** A clean, interactive web interface for document uploading, querying, and verifying sources.

## System Architecture

The pipeline processes documents through a multi-stage retrieval system before generating an answer.

```text
+-------------------+
|    PDF Upload     |
+-------------------+
          ↓
+-------------------+
| Clause Extraction |  <-- Custom Regex splits on "Section/Article"
+-------------------+
          ↓
+-------------------+
|   Embedding Gen   |  <-- all-MiniLM-L6-v2 (Sentence Transformers)
+-------------------+
          ↓
+-------------------+
| BM25 + Chroma DB  |  <-- Ensemble Retriever (Sparse + Dense)
+-------------------+
          ↓
+-------------------+
| FlashRank Rerank  |  <-- Cross-encoder compresses to Top-K chunks
+-------------------+
          ↓
+-------------------+
|     Groq LLM      |  <-- Llama-3.3-70b-versatile (Strict JSON Output)
+-------------------+
          ↓
+-------------------+
| Answer + Sources  |  <-- Rendered in UI with hash-mapped citations
+-------------------+

Technology Stack
Python: Core application logic.

Streamlit: Frontend UI and session state management.

LangChain: Orchestration framework for retrieval and RAG logic.

ChromaDB: Local vector database for dense embeddings.

Sentence Transformers: all-MiniLM-L6-v2 for lightweight, fast local embeddings.

BM25Retriever: Lexical search implementation.

FlashRank: Ultra-lightweight reranking cross-encoder.

Groq API: High-speed inference provider.

Llama 3.3 70B: Foundational LLM for compliance reasoning and JSON structuring.



Challenges & Engineering Decisions
Building a robust contract RAG system presented several engineering tradeoffs:

Character-based Chunking Causing Clause Bleeding: Initially, using RecursiveCharacterTextSplitter resulted in broken legal clauses where the context of a paragraph was severed from its section header. This was mitigated by implementing a custom Regex parser that attempts to split the text on natural structural boundaries (e.g., matching "Section 1.X").

Hybrid Retrieval vs. Vector-Only: Dense embeddings often failed to retrieve documents based on specific alphanumerics (e.g., "Net-30" or "Net-90"). Adding a BM25 sparse retriever to run in parallel with the vector store (weighted at 0.3 / 0.7) drastically improved keyword recall for specific financial terms.

Reranking Tradeoffs: Using a cross-encoder (FlashRank) adds processing latency to the pipeline. However, it was a necessary tradeoff to compress the retrieved context window and ensure the most relevant chunks are prioritized, minimizing context bloat.

Source Attribution Challenges: Getting LLMs to accurately cite their sources without hallucinating citations is difficult. The system solves this by generating an MD5 hash for each retrieved chunk before inference. The LLM is forced via system prompt to output a JSON object containing an array of used hash IDs, mapping the answer directly back to the unedited source chunk.

Real-World PDF Formatting Inconsistencies: Documents with complex headers, footers, or multi-column layouts frequently break standard PDF parsers, requiring robust fallback chunking mechanisms if the Regex clause extractor fails.

Tradeoff Between Answer Quality and Citation Granularity: Larger chunk sizes give the LLM better reasoning context but make the final source citations overly broad. The system balances this by attempting to isolate individual numbered clauses.



Evaluation & Testing
The system was evaluated against various contract archetypes to test retrieval accuracy and reasoning boundaries.

Test Corpus: Included standard Non-Disclosure Agreements (NDAs), Master Service Agreements (MSAs), Vendor Agreements, and general Procurement Contracts.

Fact Retrieval: Tested the system's ability to pull exact numbers (e.g., "What is the liability cap?") across multiple loaded documents.

Multi-Clause Reasoning: Evaluated prompts that required synthesizing information from different parts of a document (e.g., finding the termination notice period and cross-referencing it with auto-renewal clauses).

Negative Retrieval & Hallucination Resistance: Explicitly tested by querying for information completely absent from the text (e.g., querying software licensing terms against a standard NDA). The system prompt and JSON schema are engineered to return empty source arrays [] and state that the information is missing, rather than attempting to infer an answer from irrelevant chunks.

Limitations
Citation Quality Depends on Document Structure: If a PDF is poorly formatted or lacks clear section headers, the system falls back to arbitrary character chunking, which reduces the precision of the verifiable source citations.

PDF Parsing Limitations: PyPDFLoader struggles with multi-column layouts, nested tables, and scanned documents without text layers.

Not Legal Advice: This tool is a semantic search and summarization assistant. Its outputs do not constitute legal advice and must be verified by a human professional. The model can still misinterpret highly complex or contradictory legal jargon.



Future Improvements
Layout-Aware Parsing: Replacing the current PDF loader with a layout-aware parser (e.g., LayoutParser or unstructured.io) to properly index tables and multi-column formats.

OCR Support: Integrating Tesseract or a similar OCR engine to process scanned, non-text-layer contracts.

Cross-Document Analysis: Implementing a multi-query retrieval step to explicitly contrast conflicting terms between a master agreement and a sub-contract.

Multi-Agent Workflows: Utilizing an agentic framework where a "Researcher" agent gathers clauses and a "Reviewer" agent validates the findings before returning the final JSON to the user.

Better Legal Clause Attribution: Fine-tuning the chunking boundaries using a specialized legal NLP model rather than relying entirely on regular expressions.

Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/docuaudit-ai.git](https://github.com/your-username/docuaudit-ai.git)
cd docuaudit-ai
Set up a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the project root and add your Groq API key:

Code snippet
GROQ_API_KEY=your_api_key_here
Usage
Start the application:

Bash
streamlit run app.py
Navigate to http://localhost:8501 in your browser.

Upload one or multiple PDF contracts via the central uploader.

Use the predefined quick-action buttons for common audits (e.g., "Audit Financial Liabilities").

Alternatively, use the chat input to ask specific semantic questions about the uploaded documents.

Click the "Verifiable Source Documents Used" expander below any AI response to trace the answer back to the exact source text and page number.

docuaudit-ai/
│
├── app.py                 # Main Streamlit application and RAG pipeline
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (git-ignored)
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
License
This project is licensed under the MIT License. See the LICENSE file for details.
