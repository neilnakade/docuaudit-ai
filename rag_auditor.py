import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Setup Groq (Replace with your actual key)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("1. Reading and chunking the contract...")
loader = PyPDFLoader("sample_contract.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# 2. Create the Vector Database (The "Brain")
# We use a free, lightweight embedding model that runs directly on your machine
print("2. Converting text to vectors (This takes a few seconds the first time)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings)

# 3. Search for a specific risk (The "Retrieval")
query = "Under what conditions can the company terminate this employment agreement?"
print(f"\n3. Searching the database for: '{query}'")

# We ask the database to find the 3 most mathematically relevant chunks
found_docs = vector_db.similarity_search(query, k=3) 

# Combine the found text into one string
context = ""
for i, doc in enumerate(found_docs):
    context += f"--- Clause Segment {i+1} ---\n{doc.page_content}\n\n"

# 4. Send to AI (The "Augmented Generation")
print("4. Sending the relevant clauses to Groq AI for legal analysis...\n")

system_prompt = """
You are a strict legal auditor. Read the provided contract clauses and answer the user's question. 
If the answer is not in the text, say 'I cannot find this in the contract.' 
Do not make up outside legal advice. Keep your answer professional and concise.
"""

user_prompt = f"Contract Clauses:\n{context}\n\nQuestion: {query}"

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print("========== AI AUDIT RESULT ==========")
print(completion.choices[0].message.content)
print("=====================================")cf 