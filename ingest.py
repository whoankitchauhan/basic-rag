# ============================================
# STEP 1: Import required libraries
# ============================================

from pypdf import PdfReader
import ollama
import chromadb
import shutil
import os
import re


# ============================================
# STEP 2: Read the PDF file
# ============================================

# Open our PDF file
reader = PdfReader("document.pdf")

# This variable will store all extracted text
text = ""

# Go through every page in the PDF
for page in reader.pages:
    text += page.extract_text()


# ============================================
# STEP 3: Split text into sentence-aware chunks
# ============================================

def create_chunks(text, chunk_size=500, overlap_sentences=1):

    # --------------------------------------------
    # 3A. Split the text into sentences
    # --------------------------------------------

    # Split whenever we find ., !, or ?
    # followed by whitespace.
    #
    # Example:
    # "Hello. How are you?"
    #
    # becomes:
    # ["Hello.", "How are you?"]

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    # This list will contain our final chunks
    chunks = []

    # Temporarily stores sentences for
    # the chunk currently being created
    current_sentences = []


    # --------------------------------------------
    # 3B. Build chunks from complete sentences
    # --------------------------------------------

    for sentence in sentences:

        # Remove unnecessary spaces
        sentence = sentence.strip()

        # Ignore empty pieces
        if not sentence:
            continue

        # Test what the chunk would look like
        # if we added this sentence
        test_chunk = " ".join(
            current_sentences + [sentence]
        )

        # If the sentence fits within our
        # target chunk size, add it
        if len(test_chunk) <= chunk_size:

            current_sentences.append(sentence)

        else:

            # Save the current chunk
            if current_sentences:
                chunks.append(
                    " ".join(current_sentences)
                )

            # Keep the last sentence from the
            # previous chunk as overlap
            current_sentences = current_sentences[
                -overlap_sentences:
            ]

            # Add the new sentence
            current_sentences.append(sentence)


    # --------------------------------------------
    # 3C. Save the final chunk
    # --------------------------------------------

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


# Create the chunks
chunks = create_chunks(text)


# ============================================
# STEP 4: Check our chunks
# ============================================

print("\n================================")
print("Chunking Results")
print("================================")

print("Number of characters:", len(text))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i} ---")

    # Show the size of this chunk
    print("Characters:", len(chunk))

    # Show the actual chunk
    print(chunk)


# ============================================
# STEP 5: Create embeddings for all chunks
# ============================================

# This list will store the embedding vector
# for every chunk
embeddings = []

# Process every chunk one by one
for i, chunk in enumerate(chunks):

    print(f"\nCreating embedding for Chunk {i}...")

    # Send the chunk to our local
    # EmbeddingGemma model
    response = ollama.embed(
        model="embeddinggemma:latest",
        input=chunk
    )

    # Extract the first embedding
    vector = response.embeddings[0]

    # Store the vector
    embeddings.append(vector)


# ============================================
# STEP 6: Create a fresh ChromaDB database
# ============================================

# Location where ChromaDB will store its data
db_path = "./chroma_db"

# If an old database exists, delete it
# so we can rebuild it from the current PDF
if os.path.exists(db_path):

    shutil.rmtree(db_path)

    print("\nOld ChromaDB database deleted.")


# Create a new persistent ChromaDB database
client = chromadb.PersistentClient(
    path=db_path
)

print("Fresh ChromaDB database created.")


# ============================================
# STEP 7: Create ChromaDB collection
# ============================================

# Create a fresh collection for our document
collection = client.create_collection(
    name="documents"
)

print("Collection created:", collection.name)


# ============================================
# STEP 8: Store chunks and embeddings
# ============================================

# Create a unique ID for every chunk
#
# Example:
# chunk_0
# chunk_1
# chunk_2
# ...

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# Store the following together:
#
# 1. Unique ID
# 2. Embedding vector
# 3. Original text

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks
)


# ============================================
# STEP 9: Verify ChromaDB
# ============================================

print("\n================================")
print("ChromaDB Results")
print("================================")

# Number of records stored
print(
    "Number of records:",
    collection.count()
)


# Retrieve the documents and embeddings
# so we can verify that they were stored
stored_data = collection.get(
    include=[
        "documents",
        "embeddings"
    ]
)


# Show stored IDs
print(
    "Stored IDs:",
    stored_data["ids"]
)


# Show first stored document
print("\nFirst stored document:")
print(
    stored_data["documents"][0]
)


# Show first embedding size
print("\nFirst embedding length:")
print(
    len(stored_data["embeddings"][0])
)


# Show first 5 values of the first embedding
print("\nFirst 5 values of first embedding:")
print(
    stored_data["embeddings"][0][:5]
)