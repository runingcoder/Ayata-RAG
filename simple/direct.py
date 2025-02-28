import os
import time
from typing import List
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "md"
COMPANY_FILE = "Ayata.txt"

# Global instances
llm = Ollama(model="gemma:2b")
embeddings = OllamaEmbeddings(model="gemma:2b")
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
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
        existing_docs = vector_store.get()
        if existing_docs["metadatas"]:
            for meta in existing_docs["metadatas"]:
                if meta.get("source") == os.path.join(DATA_PATH, COMPANY_FILE):
                    stored_time = meta.get("last_modified", 0)
                    current_time = os.path.getmtime(os.path.join(DATA_PATH, COMPANY_FILE))
                    if stored_time >= current_time:
                        print(f"Using existing Chroma DB for {COMPANY_FILE}.")
                        return
    
    # Embed the document
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
    
    full_text = documents[0].page_content
    metadata = {
        "source": file_path,
        "last_modified": os.path.getmtime(file_path)
    }
    return Document(page_content=full_text, metadata=metadata)

def embed_document(document: Document):
    global vector_store
    if os.path.exists(CHROMA_PATH):
        try:
            import shutil
            shutil.rmtree(CHROMA_PATH)  # Clear old DB fully
        except Exception as e:
            print(f"Error clearing Chroma DB: {e}")
    
    # Use cosine similarity explicitly
    vector_store = Chroma.from_documents(
        [document], embeddings, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Saved document to {CHROMA_PATH}.")

def query_rag(question: str, relevance_threshold: float = 0.5):
    if vector_store is None:
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only): {response}\n")
        return

    # Debug: Print raw scores
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=1)
    print("Debug: Raw docs_with_scores:", docs_with_scores)
    
    if docs_with_scores:
        doc, score = docs_with_scores[0]
        print(f"Debug: Relevance score: {score}")

        if score >= relevance_threshold:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="Context: {context}\nQuestion: {question}\nAnswer:"
            )
            formatted_prompt = prompt.format(context=doc.page_content, question=question)
            response = llm.invoke(formatted_prompt)
            print(f"Query: {question}")
            print(f"Response (RAG, relevance score: {score:.2f}): {response}\n")
        else:
            response = llm.invoke(question)
            print(f"Query: {question}")
            print(f"Response (LLM only, relevance score: {score:.2f}): {response}\n")
    else:
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only, no scores returned): {response}\n")

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