from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # 2. Split the text into chunks
    # We use 1000 characters with a little overlap so sentences don't get cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)
    
    print(f"Successfully split the contract into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test it with your sample file
    test_chunks = process_pdf("sample_contract.pdf")
    print(f"First chunk preview: {test_chunks[0].page_content[:100]}...")-