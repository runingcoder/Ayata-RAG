import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "examples_data"
COMPANY_FILE = "nke-10k-2023.pdf"

# Global embedding instance
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def main():
    # Initialize and load document, then embed
    initialize_embedding()

def initialize_embedding():
    document = load_document()
    if not document:
        print("No document found.")
        return

    # Embed the document using the embeddings model
    all_splits = split_documents(document)
    embed_document(document)
    print(f"Embedding complete for {COMPANY_FILE}.")

def load_document() -> Document:
    file_path = os.path.join(DATA_PATH, COMPANY_FILE)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    add_start_index=True)
    all_splits = text_splitter.split_documents(documents)
    print(len(all_splits), 'lenght of allsplits,')
    print(type(all_splits), 'type of all splits')

    return all_splits

def embed_document(document):
    if os.path.exists(CHROMA_PATH):
        try:
            import shutil
            shutil.rmtree(CHROMA_PATH)  # Clear old DB fully
        except Exception as e:
            print(f"Error clearing Chroma DB: {e}")
    
    # vector_store = Chroma(embedding_function=embeddings)
    # ids = vector_store.add_documents(documents=document)
    vector_store = Chroma.from_documents(
        document, embeddings, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Document embedded and stored in {CHROMA_PATH}.")
    query = "Tell me about PRODUCT RESEARCH, DESIGN AND DEVELOPMENT of Nike."

# Perform similarity search
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)
    for doc, score in results:
        print(f"Relevance Score: {score:.4f}")
        print(f"Content: {doc.page_content}\n")
    return 'All done'


if __name__ == "__main__":
    main()
