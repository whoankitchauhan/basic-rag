# ============================================
# STEP 1: Import required libraries
# ============================================

import ollama
import chromadb


# ============================================
# STEP 2: Connect to our existing ChromaDB
# ============================================

# Connect to the local ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")

# Open the collection containing our document
# chunks and their embeddings
collection = client.get_collection(
    name="documents"
)


# ============================================
# STEP 3: Get the user's question
# ============================================

question = "Why are silicon spin qubits suitable for commercial scalability?"

print("Question:", question)


# ============================================
# STEP 4: Convert the question into an embedding
# ============================================

# Use the SAME embedding model that we used
# during ingestion.
response = ollama.embed(
    model="embeddinggemma:latest",
    input=question
)

# Extract the question's vector
question_embedding = response.embeddings[0]

print("\nQuestion embedding length:", len(question_embedding))


# ============================================
# STEP 5: Retrieve relevant chunks
# ============================================

# Search ChromaDB for the 2 chunks that are
# most similar to the question.
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# ============================================
# STEP 6: Extract the retrieved text
# ============================================

# results["documents"][0] contains the documents
# retrieved for our question.
retrieved_chunks = results["documents"][0]

print("\n================================")
print("Retrieved Chunks")
print("================================")

for i, document in enumerate(retrieved_chunks):

    print(f"\n--- Result {i} ---")
    print(document)

    print("Distance:", results["distances"][0][i])


# ============================================
# STEP 7: Combine retrieved chunks into context
# ============================================

# Join the retrieved chunks together.
# This will become the context given to the LLM.
context = "\n\n".join(retrieved_chunks)


# ============================================
# STEP 8: Create the RAG prompt
# ============================================

# We tell the LLM to answer using the retrieved
# information as its context.
prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================
# STEP 9: Display the prompt
# ============================================

# Print the actual prompt so we can see exactly
# what information is being sent to the LLM.
print("\n================================")
print("Prompt Sent to LLM")
print("================================")
print(prompt)


# ============================================
# STEP 10: Generate the answer
# ============================================

# Send the prompt to our local Gemma 3 LLM.
response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ============================================
# STEP 11: Display the final answer
# ============================================

answer = response.message.content

print("\n================================")
print("Final Answer")
print("================================")
print(answer)