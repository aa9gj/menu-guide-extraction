"""
MeNu Guide RAG with LangChain

Production-ready RAG implementation using LangChain framework for
querying food-compound-disease relationships from the MeNu GUIDE knowledge graph.

Features:
- HuggingFace embeddings for semantic search
- FAISS vector store with MMR retrieval for diversity
- LangChain RetrievalQA chain with OpenAI

Usage:
    export OPENAI_API_KEY="your-api-key"
    python menu_guide_rag_w_langchain.py
"""

import json
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# =============================================================================
# Configuration
# =============================================================================

CHUNKS_FILE = "rag_chunks.jsonl"
VECTORSTORE_PATH = "rag_faiss_langchain"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"

# MMR retrieval parameters
RETRIEVER_K = 5  # Number of documents to retrieve
LAMBDA_MULT = 0.7  # Higher = more relevance, lower = more diversity


# =============================================================================
# Document Loading
# =============================================================================

def load_documents(chunks_file):
    """
    Load documents from JSONL file into LangChain Document format.

    Args:
        chunks_file: Path to JSONL file with chunks

    Returns:
        List of LangChain Document objects
    """
    docs = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            docs.append(
                Document(
                    page_content=data["body"],
                    metadata={"title": data["title"], "id": data.get("id", "")}
                )
            )
    return docs


# =============================================================================
# Vector Store Functions
# =============================================================================

def create_vectorstore(docs, model_name=EMBEDDING_MODEL):
    """
    Create a FAISS vector store from documents.

    Args:
        docs: List of LangChain Document objects
        model_name: HuggingFace model name for embeddings

    Returns:
        FAISS vector store
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore


def save_vectorstore(vectorstore, path=VECTORSTORE_PATH):
    """
    Save vector store to disk.

    Args:
        vectorstore: FAISS vector store
        path: Directory path to save to
    """
    vectorstore.save_local(path)


def load_vectorstore(path=VECTORSTORE_PATH, model_name=EMBEDDING_MODEL):
    """
    Load vector store from disk.

    Args:
        path: Directory path to load from
        model_name: HuggingFace model name (must match what was used to create)

    Returns:
        FAISS vector store
    """
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )


# =============================================================================
# QA Chain Functions
# =============================================================================

def create_retriever(vectorstore, search_type="mmr", k=RETRIEVER_K, lambda_mult=LAMBDA_MULT):
    """
    Create a retriever from the vector store.

    Uses MMR (Maximum Marginal Relevance) by default for diverse results.

    Args:
        vectorstore: FAISS vector store
        search_type: "mmr" for diverse results, "similarity" for pure relevance
        k: Number of documents to retrieve
        lambda_mult: MMR diversity parameter (0=max diversity, 1=max relevance)

    Returns:
        LangChain retriever
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )


def create_qa_chain(retriever, model_name=LLM_MODEL):
    """
    Create a RetrievalQA chain for answering questions.

    Args:
        retriever: LangChain retriever
        model_name: OpenAI model name

    Returns:
        RetrievalQA chain
    """
    llm = ChatOpenAI(model=model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


def query(qa_chain, question):
    """
    Query the RAG system.

    Args:
        qa_chain: RetrievalQA chain
        question: User's question

    Returns:
        Dictionary with 'result' (answer) and 'source_documents'
    """
    return qa_chain.invoke(question)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Check if chunks file exists
    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: {CHUNKS_FILE} not found.")
        print("Please run menu_guide_chunking_prototype.py first to create chunks.")
        return

    # Load documents
    print(f"Loading documents from {CHUNKS_FILE}...")
    docs = load_documents(CHUNKS_FILE)
    print(f"Loaded {len(docs)} documents")

    # Create embeddings and vector store
    print(f"Creating vector store with {EMBEDDING_MODEL}...")
    vectorstore = create_vectorstore(docs)

    # Save for later use
    save_vectorstore(vectorstore)
    print(f"Saved vector store to {VECTORSTORE_PATH}")

    # Create retriever with MMR for diverse results
    retriever = create_retriever(vectorstore)

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set. Skipping QA chain test.")
        print("Testing retrieval only...")
        test_docs = retriever.invoke("Which foods are associated with lung cancer?")
        print(f"Retrieved {len(test_docs)} documents:")
        for doc in test_docs[:3]:
            print(f"  - {doc.metadata.get('title', 'Untitled')}")
        return

    # Create QA chain
    print("Creating QA chain...")
    qa_chain = create_qa_chain(retriever)

    # Test query
    test_question = "Which foods are associated with lung cancer?"
    print(f"\nQuery: {test_question}")

    try:
        response = query(qa_chain, test_question)
        print(f"\nAnswer: {response['result']}")
        print(f"\nSources ({len(response['source_documents'])} documents):")
        for doc in response['source_documents'][:3]:
            print(f"  - {doc.metadata.get('title', 'Untitled')}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
