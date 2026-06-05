import streamlit as st
import os
import time
from groq import Groq

# 1. BASE IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 2. SAFE ENSEMBLE IMPORT
try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

# 3. MANUAL FLASHRANK ENGINE (Rebranded)
class DocuAuditReranker:
    def __init__(self):
        self.ranker = Ranker(cache_dir=".")
    
    def compress_documents(self, query, documents):
        if not documents: return []
        passages = [{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(documents)]
        req = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(req)
        return [Document(page_content=r['text'], metadata=r['meta']) for r in results[:5]]

# --- 4. CONFIGURATION & SESSION ---
st.set_page_config(page_title="DocuAudit AI", layout="wide")

@st.cache_resource
def get_reranker_engine():
    return DocuAuditReranker()

if "db_version" not in st.session_state:
    st.session_state.db_version = 0
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrievers" not in st.session_state:
    st.session_state.retrievers = {}
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "engine" not in st.session_state:
    st.session_state.engine = get_reranker_engine()

api_key = os.environ.get("GROQ_API_KEY")

st.sidebar.title("💼 DocuAudit Settings")
st.sidebar.markdown("---")
st.sidebar.success("🔒 Cloud Gateway Secure")
st.sidebar.caption("Enterprise data isolation active.")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Workspace", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.retrievers = {}
    st.session_state.file_names = []
    st.session_state.db_version += 1        
    st.session_state.file_uploader_key += 1 
    st.rerun()


# --- 5. RAG PIPELINE ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_retriever(file_path, original_name):
    loader = PyPDFLoader(file_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
    
    # Tag each chunk with its true source filename
    for chunk in chunks:
        chunk.metadata["filename"] = original_name
    
    unique_collection = f"docuaudit_session_{st.session_state.db_version}"
    
    # Isolate vector searches strictly to this file name
    vector_retriever = Chroma.from_documents(
        chunks, 
        get_embeddings(),
        collection_name=unique_collection
    ).as_retriever(search_kwargs={
        "k": 15,
        "filter": {"filename": original_name}
    })
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 15
    
    return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])


# --- 6. UI ---
st.title("DocuAudit AI: Procurement & Compliance Engine")
st.markdown("Automated hybrid-search auditing for vendor agreements, NDAs, and corporate compliance.")

files = st.file_uploader(
    "Upload Vendor Contracts (PDF)", 
    type="pdf", 
    accept_multiple_files=True, 
    key=f"uploader_{st.session_state.file_uploader_key}"
)

if files and api_key:
    for f in files:
        if f.name not in st.session_state.file_names:
            with st.spinner(f"Indexing {f.name}..."):
                temp = f"temp_{f.name}"
                with open(temp, "wb") as buffer: buffer.write(f.getbuffer())
                st.session_state.retrievers[f.name] = build_retriever(temp, f.name)
                st.session_state.file_names.append(f.name)
                os.remove(temp)

# --- 7. ONE-CLICK PROCUREMENT ACTIONS ---
clicked_query = None
if st.session_state.file_names:
    st.markdown("### Quick Compliance Checks")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟥 Scan for Critical Vendor Risks"):
            clicked_query = "Act as a Corporate Compliance Auditor. Scan the documents for critical risks such as unilateral termination, hidden auto-renewals, or unreciprocal indemnifications. Group the findings clearly."
    with col2:
        if st.button("🟨 Audit Financial Liabilities"):
            clicked_query = "Extract and audit all financial terms, including payment timelines (e.g., Net-30), monthly retainers, late fees, and hidden penalties. Display as a clean breakdown."
    with col3:
        if st.button("⚖️ Compare Multi-Vendor Terms"):
            clicked_query = "Provide a side-by-side comparative analysis of the core business and legal terms across all uploaded vendor agreements. Output as a Markdown table."
            
    st.divider()

if st.session_state.file_names:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    typed_query = st.chat_input("Ask a specific compliance or procurement question...")
    final_query = typed_query if typed_query else clicked_query

    if final_query:
        with st.chat_message("user"): st.markdown(final_query)
        
        with st.spinner("Executing Hybrid Search & Reranking..."):
            start = time.time()
            context = ""
            for name, retriever in st.session_state.retrievers.items():
                docs = retriever.invoke(final_query)
                refined = st.session_state.engine.compress_documents(final_query, docs)
                context += f"\n--- DOCUMENT: {name} ---\n" + "\n".join([d.page_content for d in refined])

            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an elite Corporate Compliance Auditor and Procurement Specialist. "
                            "You MUST base your answers STRICTLY on the provided text. If info is missing, say 'The provided documents do not contain this information.' "
                            "When analyzing clauses, categorize them visually using these emojis:\n"
                            "🟩 [Standard] - Normal, safe business terms.\n"
                            "🟨 [Review Advised] - Unusual terms, long payment timelines, or one-sided licenses.\n"
                            "🟥 [High Risk] - Hidden penalties, auto-renewals, unilateral termination, or severe financial liability."
                        )
                    },
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {final_query}"}
                ]
            )
            ans = res.choices[0].message.content
            latency = round(time.time() - start, 2)

        with st.chat_message("assistant"):
            st.markdown(ans)
            st.caption(f"Audit completed in {latency}s")
            st.session_state.chat_history.append({"role": "user", "content": final_query})
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()

# --- 8. EXPORT REPORT ---
if st.session_state.chat_history:
    last_message = st.session_state.chat_history[-1]
    if last_message["role"] == "assistant":
        st.divider()
        st.download_button(
            label="📥 Download Official Audit Report (.txt)",
            data=last_message["content"],
            file_name="DocuAudit_Compliance_Report.txt",
            mime="text/plain"
        )