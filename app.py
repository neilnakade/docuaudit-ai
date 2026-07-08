import os
import tempfile
import hashlib
import json
import re
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(
    page_title="DocuAudit AI",
    page_icon="📜",
    layout="wide"
)

# Initialize Session State
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Settings
with st.sidebar:
    st.title("DocuAudit Settings")
    api_key = os.getenv("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    
    st.markdown("---")
    st.markdown("🟢 **Status:** Session Active")
    st.caption("Enterprise data isolation active. Processing happens in-memory.")
    
    if st.button("🔄 Reset Workspace"):
        st.session_state.retrievers = {}
        st.session_state.all_chunks = {}
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("DocuAudit AI: Procurement & Compliance Engine")
st.caption("Automated hybrid-search auditing for vendor agreements, NDAs, and corporate compliance.")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload Vendor Contracts (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# Helper: Clause-Aware Regex Chunking
def clause_aware_chunking(documents, file_name):
    chunks = []
    clause_pattern = re.compile(r'(?=\b(?:Section|Article|Clause)\s+\d+(?:\.\d+)*\b)', re.IGNORECASE)
    
    for doc in documents:
        text = doc.page_content
        page_num = doc.metadata.get("page", 0) + 1
        
        # Split text on clause patterns if present, else fallback to block
        split_texts = clause_pattern.split(text)
        for part in split_texts:
            clean_text = part.strip()
            if clean_text:
                # Generate deterministic MD5 hash ID for source attribution
                chunk_hash = hashlib.md5(f"{file_name}_{page_num}_{clean_text[:50]}".encode()).hexdigest()[:8]
                chunks.append(Document(
                    page_content=clean_text,
                    metadata={
                        "source_file": file_name,
                        "page": page_num,
                        "chunk_id": chunk_hash
                    }
                ))
    return chunks

# Helper: Build Retriever Pipeline
def build_retriever(tmp_path, file_name):
    loader = PyPDFLoader(tmp_path)
    raw_docs = loader.load()
    
    # Clause-aware chunking
    chunks = clause_aware_chunking(raw_docs, file_name)
    
    # Robust Safety Check: Verify if ANY chunk contains readable text
    has_readable_text = any(doc.page_content.strip() for doc in chunks) if chunks else False
    if not chunks or not has_readable_text:
        st.error(f"⚠️ Could not extract any readable text from {file_name}. It appears to be a scanned image or encrypted document.")
        return None, None

    # Embeddings & Chroma Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # BM25 Lexical Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    
    # Ensemble Hybrid Retriever (30% BM25, 70% Vector)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    
    # FlashRank Cross-Encoder Reranker
    compressor = FlashrankRerank(top_n=3)
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return reranker, chunks

# Process Uploaded Documents
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.retrievers:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            
            with st.spinner(f"Indexing {f.name}..."):
                retriever, chunks = build_retriever(tmp_path, f.name)
                if retriever and chunks:
                    st.session_state.retrievers[f.name] = retriever
                    st.session_state.all_chunks[f.name] = chunks
            
            os.remove(tmp_path)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Verifiable Source Documents Used"):
                for src in msg["sources"]:
                    st.markdown(f"**[{src['source_file']} - Page {src['page']}] (ID: `{src['chunk_id']}`)**")
                    st.caption(f'"{src["content"]}"')

# Quick Action Buttons
st.markdown("### Quick Compliance Audits")
col1, col2, col3 = st.columns(3)
preset_query = None
with col1:
    if st.button("🟥 Scan for Critical Vendor Risks"):
        preset_query = "List all financial liabilities, termination penalties, indemnification clauses, and unilateral terms in the document."
with col2:
    if st.button("💳 Payment & Term Audit"):
        preset_query = "What are the payment terms, invoice due dates, interest fees for late payments, and renewal deadlines?"
with col3:
    if st.button("🛡️ NDA & Data Compliance"):
        preset_query = "What are the confidentiality obligations, data protection requirements, and liability limits?"

# Chat Input
user_query = st.chat_input("Ask a question about the uploaded contracts...") or preset_query

if user_query:
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar or set GROQ_API_KEY environment variable.")
        st.stop()
        
    if not st.session_state.retrievers:
        st.warning("Please upload at least one valid text-based PDF contract to begin analysis.")
        st.stop()

    # Add user query to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant context across all active document retrievers
    all_retrieved_docs = []
    for file_name, retriever in st.session_state.retrievers.items():
        retrieved = retriever.invoke(user_query)
        all_retrieved_docs.extend(retrieved)

    # Deduplicate docs by chunk_id
    unique_docs = {}
    for doc in all_retrieved_docs:
        cid = doc.metadata.get("chunk_id")
        if cid not in unique_docs:
            unique_docs[cid] = doc
    
    final_context_docs = list(unique_docs.values())

    # Prepare context payload for LLM
    context_blocks = []
    for doc in final_context_docs:
        context_blocks.append(
            f"[ID: {doc.metadata['chunk_id']}] Document: {doc.metadata['source_file']} (Page {doc.metadata['page']})\nContent: {doc.page_content}"
        )
    formatted_context = "\n\n---\n\n".join(context_blocks)

    # System Prompt enforcing JSON structure with Source Attribution
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are DocuAudit AI, an expert legal contract and procurement auditor.
Analyze the user's request using ONLY the provided context blocks below.

CONTEXT BLOCKS:
{context}

INSTRUCTIONS:
1. Provide a direct, factual, and technical answer.
2. Rely ONLY on the provided context blocks. Do not assume or extrapolate.
3. If the answer cannot be found in the context, state that clearly and return an empty array `[]` for used_chunk_ids.
4. You MUST respond with a valid JSON object matching this exact structure:
{{
    "answer": "Your detailed answer formatted in Markdown here.",
    "used_chunk_ids": ["hash1", "hash2"]
}}
Return ONLY the raw JSON object."""),
        ("human", "{question}")
    ])

    # LLM Initialization
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0
    )

    with st.chat_message("assistant"):
        with st.spinner("Auditing contract clauses..."):
            chain = prompt_template | llm
            response = chain.invoke({"context": formatted_context, "question": user_query})
            
            # Parse JSON response
            raw_content = response.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()

            try:
                res_json = json.loads(raw_content)
                answer_text = res_json.get("answer", "No answer generated.")
                used_ids = res_json.get("used_chunk_ids", [])
            except Exception:
                answer_text = response.content
                used_ids = [doc.metadata["chunk_id"] for doc in final_context_docs]

            # Match used chunk IDs back to source documents
            used_sources = []
            for doc in final_context_docs:
                if doc.metadata["chunk_id"] in used_ids:
                    used_sources.append({
                        "source_file": doc.metadata["source_file"],
                        "page": doc.metadata["page"],
                        "chunk_id": doc.metadata["chunk_id"],
                        "content": doc.page_content
                    })

            # Render Answer & Source Citations
            st.markdown(answer_text)
            if used_sources:
                with st.expander("🔍 Verifiable Source Documents Used"):
                    for src in used_sources:
                        st.markdown(f"**[{src['source_file']} - Page {src['page']}] (ID: `{src['chunk_id']}`)**")
                        st.caption(f'"{src["content"]}"')

            # Append to Session State
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": used_sources
            })