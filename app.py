import streamlit as st
import os
import tempfile
import time
import re
import hashlib
import json

# --- LangChain & Retrieval Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LLM Import ---
from groq import Groq

# ==========================================
# 1. PAGE CONFIGURATION & STATE INIT
# ==========================================
st.set_page_config(page_title="DocuAudit Enterprise", layout="wide")

if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_version" not in st.session_state:
    st.session_state.db_version = 1

# ==========================================
# 2. CORE UTILITIES
# ==========================================
@st.cache_resource
def get_embeddings():
    """Load lightweight, high-performance local embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_reranker():
    """Initialize FlashRank for semantic compression."""
    return FlashrankRerank()

def reset_workspace():
    """Safely increments the DB session to prevent Chroma cache bleeding."""
    st.session_state.retrievers = {}
    st.session_state.file_names = []
    st.session_state.chat_history = []
    st.session_state.db_version += 1
    st.rerun()

def parse_json_safely(raw_response):
    """Safely extracts JSON from LLM output, handling markdown blocks if present."""
    # Strip markdown formatting if the model accidentally wraps the JSON
    cleaned_response = re.sub(r'^```json\n|```$', '', raw_response.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned_response)
        return parsed.get("answer", raw_response), parsed.get("used_sources", [])
    except json.JSONDecodeError:
        # Graceful fallback: return the raw text and an empty source list, preventing a crash
        return raw_response, []

# ==========================================
# 3. ENTERPRISE PARSING & RETRIEVAL PIPELINE
# ==========================================
def build_retriever(file_path, original_name):
    """Parses PDF via regex structural boundaries and builds a hybrid retriever."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Stitch pages with internal tracking markers
    full_text = ""
    for doc in docs:
        p = doc.metadata.get("page", 0) + 1
        full_text += f" [INTERNAL_PAGE_{p}] " + doc.page_content
        
    # Enterprise Regex: Matches "Section X", "Article X", "X.X", "X.", or "X "
    clause_pattern = r'(?im)(?=^\s*(?:section\s+\d+|article\s+\d+|\d+\.\d+|\d+\.?)(?:\s+[\-\:]|\s+[A-Z]))'
    raw_chunks = re.split(clause_pattern, full_text)
    
    chunks = []
    current_page = 1
    
    for raw_chunk in raw_chunks:
        text = raw_chunk.strip()
        if not text:
            continue
            
        # Extract and update internal page state
        page_markers = re.findall(r'\[INTERNAL_PAGE_(\d+)\]', text)
        if page_markers:
            current_page = int(page_markers[-1])
            
        # Clean internal markers from the final chunk
        clean_text = re.sub(r'\[INTERNAL_PAGE_\d+\]', '', text).strip()
        if len(clean_text) < 20:
            continue
            
        # Extract the True Section header for metadata (fallback to 'General')
        header_match = re.search(r'(?i)^\s*(section\s+\d+|article\s+\d+|\d+\.\d+|\d+\.?)', clean_text)
        true_section = header_match.group(1).strip().upper() if header_match else "General"
        
        chunks.append(Document(
            page_content=clean_text,
            metadata={
                "filename": original_name,
                "page": current_page - 1,
                "true_section": true_section
            }
        ))
        
    # Failsafe: If the contract has zero standard headers, fallback to character splitting
    if not chunks:
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
        for chunk in chunks:
            chunk.metadata["filename"] = original_name
            chunk.metadata["true_section"] = "General"

    # Isolate Chroma collection using session state
    unique_collection = f"docuaudit_session_{st.session_state.db_version}"
    
    # Vector Search (Top 4)
    vector_retriever = Chroma.from_documents(
        chunks, 
        get_embeddings(),
        collection_name=unique_collection
    ).as_retriever(search_kwargs={"k": 4, "filter": {"filename": original_name}})
    
    # Keyword Search (Top 4)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    
    # Hybrid Ensemble
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.3, 0.7])

# ==========================================
# 4. USER INTERFACE & SIDEBAR
# ==========================================
st.sidebar.title("🏢 DocuAudit AI")
st.sidebar.caption("Enterprise RAG with Immutable Citations")

api_key = st.sidebar.text_input("Groq API Key", type="password")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

uploaded_files = st.sidebar.file_uploader("Upload Legal Contracts (PDF)", type=["pdf"], accept_multiple_files=True)

if st.sidebar.button("🔄 Reset Workspace"):
    reset_workspace()

# Process Uploads
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.file_names:
            with st.spinner(f"Ingesting {f.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                
                st.session_state.retrievers[f.name] = build_retriever(tmp_path, f.name)
                st.session_state.file_names.append(f.name)
                os.remove(tmp_path)
    
    if "engine" not in st.session_state:
        st.session_state.engine = get_reranker()

# ==========================================
# 5. CHAT UI & HISTORY
# ==========================================
st.title("Contract Intelligence Terminal")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Verifiable Source Documents Used"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['clause']}** | File: `{src['filename']}` | **Page: {src['page']}**")
                    st.info(f'"{src["content"]}"')

# ==========================================
# 6. INFERENCE & METADATA PROVENANCE ENGINE
# ==========================================
if final_query := st.chat_input("Query your contracts (e.g., 'What are the IP rights?')"):
    
    if not st.session_state.retrievers:
        st.warning("Please upload a document first.")
        st.stop()
        
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please provide a valid Groq API Key.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(final_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing Hybrid Search & Semantic Reranking..."):
            start = time.time()
            context_prompt = ""
            source_metadata = {} # Dictionary for O(1) secure ID lookup
            
            # Retrieve & Rerank 
            for name, retriever in st.session_state.retrievers.items():
                docs = retriever.invoke(final_query)
                # Compress to top 3 clauses
                # ✨ FIXED: Sequence[Document] first, query string second
                refined = st.session_state.engine.compress_documents(docs, final_query)[:3]
                
                for doc in refined:
                    filename = doc.metadata.get("filename", "Unknown")
                    page_num = doc.metadata.get("page", 0) + 1
                    true_section = doc.metadata.get("true_section", "General")
                    
                    # 1. Generate Stable Cryptographic ID for immutable provenance
                    raw_id_string = f"{filename}_{true_section}_{page_num}_{doc.page_content[:30]}"
                    stable_id = f"doc_{hashlib.md5(raw_id_string.encode('utf-8')).hexdigest()[:8]}"
                    
                    # 2. Hardcode ground truth in backend state
                    source_metadata[stable_id] = {
                        "id": stable_id,
                        "clause": true_section,
                        "filename": filename,
                        "page": page_num,
                        "content": doc.page_content.strip()
                    }
                    
                    # 3. Build context explicitly using the ID
                    context_prompt += f"\n--- ID: {stable_id} ---\n"
                    context_prompt += f"File: {filename} | Section: {true_section}\n"
                    context_prompt += f"{doc.page_content}\n"

            # LLM System Prompt with Strict JSON Output Instructions
            system_prompt = (
                "You are an elite Corporate Compliance Auditor and Procurement Specialist. "
                "You MUST base your answers STRICTLY on the provided context documents. "
                "Categorize findings visually using: 🟩 [Standard], 🟨 [Review Advised], 🟥 [High Risk]. "
                "You do NOT need to generate inline citation brackets. "
                "CRITICAL: You must output your response purely as a valid JSON object. Do not include markdown formatting or conversational filler. "
                "The JSON must strictly match this exact schema:\n"
                "{\n"
                '  "answer": "Your detailed analysis here...",\n'
                '  "used_sources": ["doc_abc123", "doc_xyz789"]\n'
                "}\n"
                "Only include IDs in 'used_sources' that directly support your answer. Do not hallucinate IDs."
            )

            # API Call
            client = Groq()
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.0, 
                response_format={"type": "json_object"}, # Enforces JSON Mode via API
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context Documents:\n{context_prompt}\n\nQuestion: {final_query}"}
                ]
            )
            
            raw_ans = res.choices[0].message.content
            
            # Safe JSON parsing
            ans, claimed_ids = parse_json_safely(raw_ans)
            
            # Silently map and filter verified sources (Zero Trust Logic)
            verified_sources = []
            for cid in claimed_ids:
                if cid in source_metadata:
                    verified_sources.append(source_metadata[cid])
            
            latency = round(time.time() - start, 2)
            
            # Render UI
            st.markdown(ans)
            if verified_sources:
                with st.expander("🔍 Verifiable Source Documents Used"):
                    for src in verified_sources:
                        st.markdown(f"**{src['clause']}** | File: `{src['filename']}` | **Page: {src['page']}**")
                        st.info(f'"{src["content"]}"')

            st.caption(f"Audit completed in {latency}s")
            
            # Update history with structured data
            st.session_state.chat_history.append({"role": "user", "content": final_query})
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": ans, 
                "sources": verified_sources 
            })
            st.rerun()

# ==========================================
# 7. EXPORT LOGIC
# ==========================================
if st.session_state.chat_history:
    last_message = st.session_state.chat_history[-1]
    if last_message["role"] == "assistant":
        st.divider()
        
        export_text = last_message["content"]
        if last_message.get("sources"):
            export_text += "\n\n--- VERIFIABLE SOURCE CLAUSES ---\n"
            for src in last_message["sources"]:
                export_text += f"\n{src['clause']} | File: {src['filename']} | Page: {src['page']}\n\"{src['content']}\"\n"
        
        st.download_button(
            label="📥 Download Official Audit Report (.txt)",
            data=export_text,
            file_name="DocuAudit_Compliance_Report.txt",
            mime="text/plain"
        )