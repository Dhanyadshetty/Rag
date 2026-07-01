import numpy as np
import faiss
from groq import Groq
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Setup Embedding Model 

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Load Data

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 2. Chunking

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# 3. Vector Store 

def build_faiss_index(chunks):
    print(" Creating embeddings (batch)...")

    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(" FAISS index built!")

    return index, embeddings

# 4. Retrieve

def retrieve(query, index, chunks, top_k=2):
    query_embedding = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]

# 5. Generate Answer 

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
You are a smart assistant.

Rules:
- Answer ONLY from the context
- Do NOT use outside knowledge
- If not found, say "I don't know"

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def main():
    text = load_data("data.txt")   

    chunks = chunk_text(text)

    index, _ = build_faiss_index(chunks)

    print("\n RAG system ready! Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ")

        if query.lower() == "exit":
            break

        context = retrieve(query, index, chunks)
        answer = generate_answer(query, context)

        print("\n Answer:\n", answer, "\n")

if __name__ == "__main__":
    main()
