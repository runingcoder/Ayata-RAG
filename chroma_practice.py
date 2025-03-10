import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

COMPANY_FILE = 'Ayata pdf.pdf'

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/")
# collection = client.create_collection(name="Students")

student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""

# collection.add(
#     documents = [student_info, club_info, university_info],
#     metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
#     ids = ["id1", "id2", "id3"]
# )
# student_collection = collection.get_collection("Students")

# collection_name = "ayata_collection"  # You can name it whatever you want
# collection = client.get_or_create_collection(
#     name=collection_name,
#     embedding_function=embedding_model,  # Pass the embedding model here
#     metadata={"hnsw:space": "cosine", "filename": COMPANY_FILE}  # Collection metadata
# )
texts = ["Hello world!", "Machine learning is fascinating.", "ChromaDB is useful for vector storage."]

# Generate embeddings
embeddings = embedding_model.embed_documents(texts)

# Print the embeddings
for i, emb in enumerate(embeddings):
    print(f"Embedding for text {i+1}: {emb[:5]}...")



print('Hello world!')