from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Chroma Vector DB API")

# Global variables for lazy loading
_embedding_model = None
_chroma_client = None
CHROMA_PATH = "db/"

# Pydantic models for request validation
class EmbedRequest(BaseModel):
    file_path: str
    collection_name: str

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    k: int = 5

class QueryOnlyRequest(BaseModel):
    query: str

def get_llm():
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:1.5b")
    logger.info("ChromaDB client initialized")
    return llm

# Lazy loading functions
def get_embedding_model():
    """Lazy load the embedding model only when needed"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Initializing embedding model...")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model initialized")
    return _embedding_model

def get_chroma_client():
    """Lazy load the ChromaDB client only when needed"""
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initializing ChromaDB client...")
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        logger.info("ChromaDB client initialized")
    return _chroma_client

def get_chroma_db(collection_name=None):
    """Lazy load a Chroma vector store with the specified collection"""
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_model()
    )

def check_existing_embeddings(collection_name: str) -> Optional[object]:
    try:
        if collection_name is None:
            collection_name = 'Ayata_doc'
        client = get_chroma_client()
        collections = client.list_collections()
        for collection in collections:
            if collection == collection_name:
                vector_store = get_chroma_db(collection_name)
                logger.info(f"Collection '{collection_name}' already exists.")
                return vector_store
    except Exception as e:
        logger.error(f"Error checking existing collection: {e}")
    logger.info(f"Collection '{collection_name}' does not exist.")
    return None

def load_document(file_path: str) -> Optional[List[object]]:
    try:
        # Lazy import
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents:
            logger.warning(f"No documents found in {file_path}")
            return None
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading document from {file_path}: {e}")
        return None

def split_documents(documents: List[object]) -> List[object]:
    # Lazy import
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(all_splits)} chunks")
    return all_splits

def embed_document(documents: List[object], collection_name: str) -> object:
    try:    
        from langchain_chroma import Chroma
        
        vector_store = Chroma.from_documents(
            documents, 
            get_embedding_model(), 
            persist_directory=CHROMA_PATH, 
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=collection_name
        )
        logger.info(f"Created collection '{collection_name}' and saved embeddings in {CHROMA_PATH}")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating and embedding documents: {e}")
        raise

# API Endpoints

@app.post("/embed", summary="Embed a PDF into a Chroma collection")
async def embed_pdf(request: EmbedRequest):
    vector_store = check_existing_embeddings(request.collection_name)
    if vector_store is not None:
        return {"message": f"Collection '{request.collection_name}' already exists."}

    documents = load_document(request.file_path)
    if documents is None:
        raise HTTPException(status_code=400, detail=f"Failed to load document from {request.file_path}")

    split_docs = split_documents(documents)
    vector_store = embed_document(split_docs, request.collection_name)
    return {"message": f"Successfully embedded {len(split_docs)} chunks into collection '{request.collection_name}'"}

@app.post("/query", summary="Query a Chroma collection for similar documents")
async def query_collection(request: QueryRequest):
    try:
        vector_store = check_existing_embeddings(request.collection_name)
        if vector_store is None:
            return {"message": f"Collection '{request.collection_name}' doesn't exist yet."}

        results = vector_store.similarity_search_with_relevance_scores(
            query=request.query,
            k=request.k
        )
        response = [
            {
                "document": doc.page_content,
                "score": score,
                "id": doc.metadata.get("id", "unknown")
            }
            for doc, score in results
        ]
        return {"results": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying collection: {str(e)}")
    

@app.post("/main_query", summary="Query a Chroma collection for llm generated answer!")
async def query_collection(request: QueryOnlyRequest, relevance_threshold : float = 0.5):
    
    vector_store = check_existing_embeddings(None)
    question = request.query
    llm =get_llm()
    if vector_store is None:
            response = llm.invoke(question)
            logger.info(f"Query: {question}")
            logger.info(f"Response (LLM only): {response}\n")
            return
    from langchain.prompts import PromptTemplate
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(
            query=question,
            k=3
        )        
    if docs_with_scores:
        # takes the doc with the highest score for now. Will do rag chaining later for summary of summary docs.
        doc, score = max(docs_with_scores, key=lambda x: x[1])
        print(f"Debug: Relevance score: {score}")

        if score >= relevance_threshold:
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
            formatted_prompt = prompt.invoke({"context": doc.page_content, "question": question}).to_messages()
            assert len(formatted_prompt) == 1, "Expected a single message from prompt"
            response = llm.invoke(formatted_prompt[0].content)
            response = response.content
            logger.info(f"Query: {question}")
            logger.info(f"Response (RAG, relevance score: {score:.2f}): {response}\n")
        else:
            logger.info('reaches here! ')
            response = llm.invoke(question)
            logger.info(f"Query: {question}")
            logger.info(f"Response (LLM only, relevance score: {score:.2f}): {response}\n")
    else:
        response = llm.invoke(question)
        logger.info(f"Query: {question}")
        logger.info(f"Response (LLM only, no scores returned): {response}\n")

    return {"messsage": f"The response: {response}" }

@app.delete("/delete-all-collections", summary="Delete all existing collections in Chroma DB")
async def delete_all_collections():
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        if not collections:
            return {"message": "No collections found to delete."}

        num_collections = len(collections)
        for collection in collections:
            client.delete_collection(name=collection)
            logger.info(f"Deleted collection update'{collection}'")

        return {"message": f"Successfully deleted {num_collections} collection(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collections: {str(e)}")

# If running directly, use this main block
if __name__ == "__main__":
    import uvicorn
    # Set reload=True during development, False in production
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)