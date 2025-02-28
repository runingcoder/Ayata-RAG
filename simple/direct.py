import os
import shutil
from typing import List
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.docstore.document import Document

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "simple/md"

# Global instances
llm = Ollama(model="gemma:2b")
embeddings = OllamaEmbeddings(model="gemma:2b")
vector_store = None
rag_chain = None

def main():
    generate_data_store()
    run_interactive_mode()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found. Running in LLM-only mode.")
        return
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> List[Document]:
    loader = DirectoryLoader(DATA_PATH, glob="*.md", show_progress=True)  # Using *.txt, adjust to *.md if needed
    documents = loader.load()
    return documents

def split_text(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        document = chunks[min(10, len(chunks) - 1)]  # Avoid index error if <10 chunks
        print("Sample chunk content:")
        print(document.page_content)
        print("Sample chunk metadata:")
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: List[Document]):
    global vector_store, rag_chain
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    # No need for db.persist() in newer versions; Chroma auto-persists

    # Set up retriever and RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_rag(question: str, relevance_threshold: float = 0.7):
    if vector_store is None or rag_chain is None:
        # No documents loaded, use LLM directly
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only): {response}\n")
        return

    # Search for relevant documents with similarity scores
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=3)
    
    # Check if the top result is relevant enough
    if docs_with_scores and docs_with_scores[0][1] >= relevance_threshold:
        # Use RAG with retrieved context
        response = rag_chain.invoke(question)
        print(f"Query: {question}")
        print(f"Response (RAG): {response['result']}\n")
    else:
        # Fallback to LLM without document context
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only, no relevant context found): {response}\n")

def run_interactive_mode():
    if vector_store is None:
        print("No documents loaded. Running in LLM-only mode.")
    else:
        print("RAG system ready with documents.")
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