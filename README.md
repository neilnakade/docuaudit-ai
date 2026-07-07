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
