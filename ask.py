# ============================================
# STEP 1: Import required libraries
# ============================================

import ollama
import chromadb


# ============================================
# STEP 2: Connect to our existing ChromaDB
# ============================================

# Connect to the local ChromaDB database
# that we created during ingestion.
client = chromadb.PersistentClient(path="./chroma_db")

# Open the same collection where our
# document chunks and embeddings are stored.
collection = client.get_collection(
    name="documents"
)


# ============================================
# STEP 3: Define the user's question
# ============================================

question = "Why are silicon spin qubits suitable for commercial scalability?"

print("Question:", question)


# ============================================
# STEP 4: Convert the question into an embedding
# ============================================

# We use the SAME embedding model that we used
# when storing our document chunks.
response = ollama.embed(
    model="embeddinggemma:latest",
    input=question
)

# Get the question's 768-dimensional vector
question_embedding = response.embeddings[0]

print("\nQuestion embedding length:", len(question_embedding))


# ============================================
# STEP 5: Search ChromaDB
# ============================================

# Ask ChromaDB to find the 2 chunks that are
# most similar to our question.
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# ============================================
# STEP 6: Display the retrieved chunks
# ============================================

print("\n================================")
print("Retrieved Chunks")
print("================================")

for i, document in enumerate(results["documents"][0]):

    print(f"\n--- Result {i} ---")
    print(document)

    # ChromaDB also gives us a distance score.
    print("Distance:", results["distances"][0][i])