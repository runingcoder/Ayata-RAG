from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from contextlib import asynccontextmanager

# Initialize app
app = FastAPI()

# Global variables to avoid reloading
llm = Ollama(model="gemma:2b")
embeddings = OllamaEmbeddings(model="gemma:2b")
vector_store = None
rag_chain = None

# Setup RAG system function
def setup_rag():
    global vector_store, rag_chain
    # Load data from file
    with open("data.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Split into smaller chunks for larger data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents([raw_text])

    # Create or load vector store (only once)
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Top 3 chunks

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

# Using lifespan event handler for startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run during startup
    setup_rag()  # Calling the synchronous setup function
    yield  # Allow FastAPI to proceed with handling requests

# Pass lifespan to FastAPI instance
app = FastAPI(lifespan=lifespan)

# API endpoint to query the RAG system
@app.get("/query/{question}")
async def query_rag(question: str):
    if rag_chain is None:
        return {"error": "RAG system not initialized"}
    response = rag_chain.invoke(question)
    return {"query": question, "response": response["result"]}

