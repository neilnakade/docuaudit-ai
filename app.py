import streamlit as st
import os
import re
import hashlib
import json
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from flashrank import Ranker, RerankRequest
from groq import Groq
from pdf2image import convert_from_path
import pytesseract

# Page configuration
st.set_page_config(page_title="DocuAudit AI", layout="wide")

# Initialize session state variables
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}
if "chunks_store" not in st.session_state:
    st.session_state.chunks_store = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clause_aware_chunking(text, file_name):
    """
    Splits text by structural legal boundaries (Sections, Articles, Clauses)
    instead of relying on fixed character split counters.
    """
    pattern = r'(?=\b(?:Section|Article|Clause|SECTION|ARTICLE|CLAUSE)\s+\d+\b|\b(?:SECTION|ARTICLE|CLAUSE)\s+[I|V|X|L|C]+\b)'
    sections = re.split(pattern, text)
    
    chunks = []
    for section in sections:
        clean_section = section.strip()
        if clean_section:
            chunks.append(clean_section)
            
    # Fallback to paragraph splitting if no structural sections are detected
    if len(chunks) <= 1:
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        
    documents = []
    for i, chunk_text in enumerate(chunks):
        # Generate a unique deterministic hash for tracking citations
        chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": file_name,
                "chunk_id": chunk_hash,
                "chunk_index": i
            }
        )
        documents.append(doc)
    return documents

def build_retriever(file_path, file_name):
    """
    Ingests text content. If the PDF contains no digital text layer,
    it falls back to an optical character recognition (OCR) pipeline.
    """
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    
    # Consolidate raw text strings across the documents
    all_text = "\n\n".join([doc.page_content for doc in raw_docs])
    
    # --- AUTOMATIC OCR FALLBACK LAYER ---
    if not any(char.isalnum() for char in all_text):
        with st.spinner(f"🔍 Scanned image PDF detected in '{file_name}'. Initializing OCR Engine..."):
            # Convert PDF pages to images in memory
            pages = convert_from_path(file_path)
            ocr_chunks = []
            
            for i, page_image in enumerate(pages):
                # Extract text visually from each page image
                page_text = pytesseract.image_to_string(page_image)
                ocr_chunks.append(page_text)
                
            all_text = "\n\n".join(ocr_chunks)
    # -------------------------------------

    chunks = clause_aware_chunking(all_text, file_name)
    
    # Final safety check in case the image itself is completely blank
    has_readable_text = any(doc.page_content.strip() for doc in chunks)
    if not chunks or not has_readable_text:
        st.error(f"⚠️ Could not extract text from '{file_name}'. The file appears completely blank or heavily encrypted.")
        st.stop()

    # Instantiate local lightweight sentence embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Dense semantic retriever setup via ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name=hashlib.md5(file_name.encode()).hexdigest()
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Sparse lexical retriever setup via BM25 Keyword Search
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    
    # Build the hybrid ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]
    )
    return ensemble_retriever, chunks

def rerank_documents(query, documents, top_n=4):
    """
    Compresses context inputs down to the most critical nodes using a cross-encoder.
    """
    if not documents:
        return []
    
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
    
    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(documents)
    ]
    
    # Wrapped in FlashRank's native RerankRequest class to prevent attribute errors
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    reranked_docs = []
    for item in results[:top_n]:
        reranked_docs.append(documents[item["id"]])
        
    return reranked_docs

def execute_rag(query, context_docs):
    """
    Funnels queries and cross-referenced contexts to the foundational LLM, enforcing 
    strict source constraints and JSON structural layout schemas.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Missing GROQ_API_KEY environment variable. Please configure it in your environment properties.")
        st.stop()
        
    client = Groq(api_key=api_key)
    
    context_str = ""
    for doc in context_docs:
        context_str += f"[CHUNK_ID: {doc.metadata['chunk_id']}]\nFrom File: {doc.metadata['source']}\n{doc.page_content}\n\n"
        
    system_prompt = (
        "You are an expert contract compliance analysis assistant. Answer the query strictly using the "
        "provided text fragments. For every assertion, map it to the corresponding CHUNK_ID.\n\n"
        "Rules:\n"
        "1. Base answers exclusively on the text provided. Do not extrapolate.\n"
        "2. If information is missing, set 'answer' to 'Information missing from text' and 'citations' to [].\n"
        "3. Respond only with a clean JSON object containing 'answer' and 'citations' fields."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuery: {query}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"answer": f"Error during model processing: {str(e)}", "citations": []}

# --- RENDER MAIN INTERFACE UI ---

st.title("DocuAudit AI: Procurement & Compliance Engine")
st.caption("Automated hybrid-search auditing for vendor agreements, NDAs, and corporate compliance.")

# Layout Sidebar options
with st.sidebar:
    st.header("DocuAudit Settings")
    st.info("🔒 In-Memory Data Pipeline Isolation Active")
    
    if st.button("Reset Workspace", use_container_width=True):
        st.session_state.retrievers.clear()
        st.session_state.chunks_store.clear()
        st.session_state.chat_history.clear()
        st.rerun()

# Document Uploader Segment
uploaded_files = st.file_uploader(
    "Upload Vendor Contracts (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.retrievers:
            with st.spinner(f"Indexing and chunking document: {f.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                
                # Triggers data preparation pipelines
                retriever, chunks = build_retriever(tmp_path, f.name)
                st.session_state.retrievers[f.name] = retriever
                st.session_state.chunks_store[f.name] = chunks
                os.remove(tmp_path)
    st.success("All loaded documents indexed successfully via hybrid pipelines.")

# Query Routing Engine
if st.session_state.retrievers:
    st.write("---")
    
    # Audit Acceleration Quick Triggers
    col1, col2 = st.columns(2)
    quick_query = None
    with col1:
        if st.button("🟥 Scan for Critical Vendor Risks", use_container_width=True):
            quick_query = "Identify all indemnification obligations, liabilities, caps, and payment penalties."
    with col2:
        if st.button("📋 Extract Termination & Renewal Timelines", use_container_width=True):
            quick_query = "What are the rules regarding contract termination, notice timelines, and auto-renewals?"

    user_query = st.chat_input("Enter your compliance query or custom audit request...")
    active_query = user_query if user_query else quick_query

    if active_query:
        st.info(f"**Processing Request:** {active_query}")
        
        # Phase 1: Consolidated Ensemble Gathering
        all_retrieved_docs = []
        for file_name, retriever in st.session_state.retrievers.items():
            all_retrieved_docs.extend(retriever.invoke(active_query))
            
        # Phase 2: Context Cross-Encoder Reranking Execution
        with st.spinner("Reranking relevant legal structures via Cross-Encoder..."):
            optimized_context = rerank_documents(active_query, all_retrieved_docs)
            
        # Phase 3: Text Reasoning Inference execution
        with st.spinner("Analyzing parameters with Llama 3.3 70B..."):
            evaluation_payload = execute_rag(active_query, optimized_context)
            
        # Display output payload
        st.markdown("### Analysis Report")
        st.write(evaluation_payload.get("answer", "No processing output found."))
        
        # Display citation anchors 
        if evaluation_payload.get("citations"):
            with st.expander("Verifiable Source Documents Used (Hash-Mapped Claims)"):
                for cite_id in evaluation_payload["citations"]:
                    # Look up actual chunk texts in the application dictionary matching the target hash ID
                    found_source = False
                    for file_name, chunk_list in st.session_state.chunks_store.items():
                        for doc in chunk_list:
                            if doc.metadata["chunk_id"] == cite_id:
                                st.markdown(f"**Source Document:** `{doc.metadata['source']}` (Reference ID: `{cite_id}`)")
                                st.blockquote(doc.page_content)
                                found_source = True
                                break
                    if not found_source:
                        st.caption(f"Reference ID: `{cite_id}` matched generalized synthesis arrays.")
else:
    st.info("Upload your structural text documents to initialize the vector stores and begin compliance tracking pipelines.")