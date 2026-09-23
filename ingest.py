# ============================================
# STEP 1: Import required libraries
# ============================================

from pypdf import PdfReader
import ollama
import chromadb
import shutil
import os


# ============================================
# STEP 2: Read the PDF file
# ============================================

# PdfReader opens our PDF file
reader = PdfReader("document.pdf")

# This variable will contain all extracted text
text = ""

# Go through every page in the PDF
for page in reader.pages:
    text += page.extract_text()


# ============================================
# STEP 3: Split the text into chunks
# ============================================

def create_chunks(text, chunk_size=500, overlap=50):

    # This list will store all our chunks
    chunks = []

    # Starting character position
    start = 0

    # Continue until we reach the end of the text
    while start < len(text):

        # Calculate where the current chunk should end
        end = start + chunk_size

        # Extract the current chunk
        chunk = text[start:end]

        # Store the chunk in our list
        chunks.append(chunk)

        # Move forward, but keep 50 characters
        # from the previous chunk as overlap
        start = end - overlap

    return chunks


# Create our chunks
chunks = create_chunks(text)


# ============================================
# STEP 4: Create embeddings for all chunks
# ============================================

# This list will store the embedding vector
# for every chunk
embeddings = []

# Go through every chunk
for i, chunk in enumerate(chunks):

    print(f"\nCreating embedding for Chunk {i}...")

    # Send the chunk to the local EmbeddingGemma model
    response = ollama.embed(
        model="embeddinggemma:latest",
        input=chunk
    )

    # Get the first embedding from the response
    vector = response.embeddings[0]

    # Store the vector
    embeddings.append(vector)





# ============================================
# STEP 5: Create a fresh ChromaDB database
# ============================================

# Path where our ChromaDB database is stored
db_path = "./chroma_db"

# If an old database already exists,
# completely remove it.
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("\nOld ChromaDB database deleted.")

# Create a completely fresh ChromaDB database
client = chromadb.PersistentClient(path=db_path)

print("Fresh ChromaDB database created.")

# ============================================
# STEP 6: Create ChromaDB collection
# ============================================

# Create a fresh collection for our document
collection = client.create_collection(
    name="documents"
)

print("Collection created:", collection.name)

# ============================================
# STEP 7: Store chunks and embeddings
# ============================================

# Create a unique ID for every chunk.
# Example:
# chunk_0, chunk_1, chunk_2...
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Store the IDs, embeddings, and original text
# together in ChromaDB.
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks
)


# ============================================
# STEP 8: Verify the stored data
# ============================================

print("\n================================")
print("ChromaDB Results")
print("================================")

# Check how many records are stored.
print("Number of records:", collection.count())

# Retrieve documents and embeddings
# so we can verify that they were stored.
stored_data = collection.get(
    include=["documents", "embeddings"]
)

print("Stored IDs:", stored_data["ids"])

print("\nFirst stored document:")
print(stored_data["documents"][0])

print("\nFirst embedding length:")
print(len(stored_data["embeddings"][0]))

print("\nFirst 5 values of first embedding:")
print(stored_data["embeddings"][0][:5])