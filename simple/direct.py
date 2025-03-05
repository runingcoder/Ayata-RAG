import os
import time
from typing import List
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain.docstore.document import Document

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "new\examples_data"
# COMPANY_FILE = "nke-10k-2023.pdf"
COMPANY_FILE = 'Ayata pdf.pdf'
# FILE_PATH = 'new\examples_data\nke-10k-2023.pdf'
FILE_PATH = 'new\examples_data\Ayata pdf.pdf'

# Global instances
llm = ChatOllama(model="zephyr:latest")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = None

def main():
    initialize_rag()
    run_interactive_mode()

def initialize_rag():
    global vector_store
    document = load_document()
    print('doc loaded')
    all_splits = split_documents(document)
    print('doc splitted')
    if not document:
        print("No document found. Running in LLM-only mode.")
        return    
    # Embed the document
    embed_document(all_splits)
    print('doc embedded')
    print(f"Initialized RAG with {COMPANY_FILE}.")

def load_document() -> Document:
    file_path = os.path.join(DATA_PATH, COMPANY_FILE)
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        return None
    print(type(documents), 'type of document')
    print(len(documents), 'length of document')
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

def embed_document(documents):
    global vector_store
    vector_store = Chroma.from_documents(
        documents, embedding_model, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Saved document to {CHROMA_PATH}.")

def query_rag(question: str, relevance_threshold: float = 0.5):
    if vector_store is None:
        response = llm.invoke(question)
        print(f"Query: {question}")
        print(f"Response (LLM only): {response}\n")
        return

    # Debug: Print raw scores
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=2)
    
    # print("Debug: Raw docs_with_scores:", docs_with_scores)
    
    if docs_with_scores:
        # takes the doc with the highest score for now. Will do rag chaining later for summary of summary docs.
        doc, score = max(docs_with_scores, key=lambda x: x[1])

        # print(f"Debug: Top document: {doc}")
        print(f"Debug: Relevance score: {score}")

        if score >= relevance_threshold:
            # Define the prompt template with your specified structure
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="You are an assistant for question-answering tasks. "
                         "Use the following pieces of retrieved context to answer the question. "
                         "If you don't know the answer, just say that you don't know. "
                         "Use as much sentences maximum as needed but keep it under 1000 words.\n"
                         "Question: {question}\n"
                         "Context: {context}\n"
                         "Answer:"
            )
            # Format the prompt and convert to messages
            formatted_prompt = prompt.invoke({"context": doc.page_content, "question": question}).to_messages()
            assert len(formatted_prompt) == 1, "Expected a single message from prompt"
            response = llm.invoke(formatted_prompt[0].content)
            print(type(response), 'type of response')
            print(f"Query: {question}")
            print(f"Response (RAG, relevance score: {score:.2f}): {response.content}\n")
        else:
            print('reaches here! ')
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