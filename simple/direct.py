import os
import time
from typing import List
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "md"
COMPANY_FILE = "company_x.txt"  # Specific file to check

# Global instances
llm = Ollama(model="zephyr:latest")
embeddings = OllamaEmbeddings(model="zephyr:latest")
vector_store = None

def main():
    initialize_rag()
    run_interactive_mode()

def initialize_rag():
    global vector_store
    document = load_document()
    if not document:
        print("No document found. Running in LLM-only mode.")
        return
    
    # Check if Chroma DB exists and contains the document
    if os.path.exists(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        existing_docs = vector_store.get()  # Get metadata of stored docs
        if existing_docs["metadatas"]:
            for meta in existing_docs["metadatas"]:
                if meta.get("source") == os.path.join(DATA_PATH, COMPANY_FILE):
                    # Check if file hasn’t changed
                    stored_time = meta.get("last_modified", 0)
                    current_time = os.path.getmtime(os.path.join(DATA_PATH, COMPANY_FILE))
                    if stored_time >= current_time:
                        print(f"Using existing Chroma DB for {COMPANY_FILE}.")
                        return
    
    # If Chroma doesn’t exist or file changed, embed the document
    embed_document(document)
    print(f"Initialized RAG with {COMPANY_FILE}.")

def load_document() -> Document:
    file_path = os.path.join(DATA_PATH, COMPANY_FILE)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    loader = DirectoryLoader(DATA_PATH, glob=COMPANY_FILE, show_progress=True)
    documents = loader.load()
    if not documents:
        return None
    
    # Combine into one document (no chunking for now)
    full_text = documents[0].page_content
    metadata = {
        "source": file_path,
        "last_modified": os.path.getmtime(file_path)
    }
    return Document(page_content=full_text, metadata=metadata)

def embed_document(document: Document):
    global vector_store
    # Clear existing DB if it exists
    if os.path.exists(CHROMA_PATH):
        os.rmdir(CHROMA_PATH)  # Use os.rmdir instead of shutil.rmtree to avoid recursive delete
    
    # Create new Chroma DB with the single document
    vector_store = Chroma.from_documents(
        [document], embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved document to {CHROMA_PATH}.")

def query_rag(question: str, relevance_threshold: float = 0.3):
    if vector_store is None:
        # No document loaded, use LLM directly
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only): {response}\n")
        return

    # Search for relevance
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=1)  # k=1 since one doc
    
    # Check if the document is relevant
    if docs_with_scores and docs_with_scores[0][1] >= relevance_threshold:
        doc = docs_with_scores[0][0]  # The single document
        # Custom prompt with full document as context
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Context: {context}\nQuestion: {question}\nAnswer:"
        )
        formatted_prompt = prompt.format(context=doc.page_content, question=question)
        response = llm.invoke(formatted_prompt)
        print(f"Query: {question}")
        print(f"Response (RAG, relevance score: {docs_with_scores[0][1]:.2f}): {response}\n")
    else:
        # Fallback to LLM without context
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only, relevance score: {docs_with_scores[0][1] if docs_with_scores else 'N/A'}): {response}\n")

def run_interactive_mode():
    if vector_store is None:
        print("No document loaded. Running in LLM-only mode.")
    else:
        print("RAG system ready with company data.")
    print("Type your question (or 'exit' to quit):")
    
    while True:
        question = input("> ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question:
            query_rag(question)

if __name__ == "__main__":
    main()