"""
MeNu Guide RAG Chunking and Retrieval Prototype

This module implements a custom RAG (Retrieval-Augmented Generation) pipeline
using FAISS for vector similarity search and OpenAI for response generation.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python menu_guide_chunking_prototype.py
"""

import json
import os
import pickle
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "final_sentences.txt"
CHUNKS_FILE = "rag_chunks.jsonl"
INDEX_FILE = "rag_faiss.index"
LOOKUP_FILE = "doc_lookup.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4"
DEFAULT_TOP_K = 5

# =============================================================================
# Global State (initialized during main execution)
# =============================================================================

model = None
index = None
doc_lookup = {}


# =============================================================================
# Chunking Functions
# =============================================================================

def create_chunks_from_sentences(input_file):
    """
    Create RAG chunks from extracted sentences.

    Each chunk contains:
    - id: Unique UUID
    - title: Extracted compound + disease combination
    - body: Full sentence text

    Args:
        input_file: Path to file containing sentences (one per line)

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            if not sentence:
                continue

            # Extract title from compound + disease if possible
            title = _extract_title(sentence)

            chunks.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "body": sentence
            })

    return chunks


def _extract_title(sentence):
    """
    Extract a title from a sentence by finding compound and disease mentions.

    Args:
        sentence: Input sentence

    Returns:
        Title string in format "Compound in Disease" or default fallback
    """
    try:
        compound = sentence.split("contains")[1].split(",")[0].strip().capitalize()
        disease = sentence.split("biomarker in")[1].split(".")[0].strip().capitalize()
        return f"{compound} in {disease}"
    except (IndexError, ValueError, AttributeError):
        return "Food-compound-disease relationship"


def save_chunks(chunks, output_file):
    """
    Save chunks to JSONL format.

    Args:
        chunks: List of chunk dictionaries
        output_file: Path to output JSONL file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def load_chunks(input_file):
    """
    Load chunks from JSONL format.

    Args:
        input_file: Path to input JSONL file

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            chunks.append(doc)
    return chunks


# =============================================================================
# Embedding and Indexing Functions
# =============================================================================

def create_embeddings(chunks, model_name=EMBEDDING_MODEL):
    """
    Create embeddings for document chunks.

    Args:
        chunks: List of chunk dictionaries (must have 'body' field)
        model_name: Name of sentence-transformers model to use

    Returns:
        Tuple of (SentenceTransformer model, numpy array of embeddings)
    """
    embedding_model = SentenceTransformer(model_name)
    corpus = [doc["body"] for doc in chunks]
    embeddings = embedding_model.encode(corpus, show_progress_bar=True)
    return embedding_model, np.array(embeddings)


def build_faiss_index(embeddings):
    """
    Build a FAISS index from embeddings.

    Uses L2 (Euclidean) distance for similarity.

    Args:
        embeddings: Numpy array of embeddings

    Returns:
        FAISS index
    """
    embedding_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(embeddings)
    return faiss_index


def save_index(faiss_index, chunks, index_file=INDEX_FILE, lookup_file=LOOKUP_FILE):
    """
    Save FAISS index and document lookup to disk.

    Args:
        faiss_index: FAISS index to save
        chunks: List of chunk dictionaries
        index_file: Path for FAISS index
        lookup_file: Path for document lookup pickle
    """
    faiss.write_index(faiss_index, index_file)

    lookup = {i: chunks[i] for i in range(len(chunks))}
    with open(lookup_file, "wb") as f:
        pickle.dump(lookup, f)


def load_index(index_file=INDEX_FILE, lookup_file=LOOKUP_FILE):
    """
    Load FAISS index and document lookup from disk.

    Args:
        index_file: Path to FAISS index
        lookup_file: Path to document lookup pickle

    Returns:
        Tuple of (FAISS index, document lookup dictionary)
    """
    faiss_index = faiss.read_index(index_file)

    with open(lookup_file, "rb") as f:
        lookup = pickle.load(f)

    return faiss_index, lookup


# =============================================================================
# Retrieval Functions
# =============================================================================

def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Retrieve most relevant documents for a query.

    Args:
        query: Query string
        top_k: Number of documents to retrieve

    Returns:
        List of document dictionaries
    """
    global model, index, doc_lookup

    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    return [doc_lookup[i] for i in indices[0]]


def build_prompt(query, retrieved_docs):
    """
    Build a RAG prompt with context and query.

    Args:
        query: User's question
        retrieved_docs: List of retrieved document dictionaries

    Returns:
        Formatted prompt string
    """
    context = "\n\n".join([doc["body"] for doc in retrieved_docs])
    return f"""Use the following biomedical information to answer the question below.

Context:
{context}

Question: {query}
Answer:"""


# =============================================================================
# LLM Query Functions
# =============================================================================

def answer_query(query, top_k=DEFAULT_TOP_K):
    """
    Answer a query using RAG pipeline with OpenAI.

    Retrieves relevant documents, builds a prompt, and generates
    a response using the configured LLM.

    Args:
        query: User's question
        top_k: Number of documents to retrieve for context

    Returns:
        Generated answer string
    """
    docs = retrieve(query, top_k)
    prompt = build_prompt(query, docs)

    client = OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function to build and test the RAG pipeline."""
    global model, index, doc_lookup

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("Please run menu_guide_neutralization.py first to generate sentences.")
        return

    # Step 1: Create chunks
    print(f"Creating chunks from {INPUT_FILE}...")
    chunks = create_chunks_from_sentences(INPUT_FILE)
    print(f"Created {len(chunks)} chunks")

    # Step 2: Save chunks
    save_chunks(chunks, CHUNKS_FILE)
    print(f"Saved chunks to {CHUNKS_FILE}")

    # Step 3: Create embeddings
    print(f"Creating embeddings with {EMBEDDING_MODEL}...")
    model, embeddings = create_embeddings(chunks)
    print(f"Created embeddings with shape {embeddings.shape}")

    # Step 4: Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Step 5: Save index and lookup
    doc_lookup = {i: chunks[i] for i in range(len(chunks))}
    save_index(index, chunks)
    print(f"Saved index to {INDEX_FILE} and lookup to {LOOKUP_FILE}")

    # Step 6: Test retrieval (if OpenAI key is available)
    if os.environ.get("OPENAI_API_KEY"):
        print("\nTesting query...")
        test_query = "Which foods are associated with lung cancer?"
        print(f"Query: {test_query}")
        try:
            answer = answer_query(test_query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error querying LLM: {e}")
    else:
        print("\nSkipping LLM test (OPENAI_API_KEY not set)")
        print("Testing retrieval only...")
        test_query = "Which foods are associated with lung cancer?"
        docs = retrieve(test_query)
        print(f"Query: {test_query}")
        print(f"Retrieved {len(docs)} documents:")
        for doc in docs[:3]:
            print(f"  - {doc['title']}")


if __name__ == "__main__":
    main()
