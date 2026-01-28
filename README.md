# MeNu Guide Knowledge Graph RAG System

A Retrieval-Augmented Generation (RAG) system for extracting and querying food-compound-disease relationships from the MeNu GUIDE knowledge graph.

## Overview

[MeNu GUIDE](https://github.com/LipiTUM/MeNuGUIDE) is a knowledge graph containing over 25 million triple statements describing relationships between foods, metabolites, and diseases. This project builds a RAG system to help clinical nutritionists and scientists query these relationships using natural language, without requiring coding expertise.

## Features

- **RDF/Turtle Processing**: Extract and transform triples from the MeNu GUIDE knowledge graph
- **Entity Enrichment**: Convert RDF entities to human-readable sentences with contextual data (amounts, units, cohorts, references)
- **Semantic Search**: FAISS-based vector similarity search using sentence embeddings
- **RAG Pipeline**: Query the knowledge base using natural language with LLM-powered responses
- **Multiple Implementations**: Both custom FAISS implementation and LangChain-based approach

## Project Structure

```
menu-guide-extraction/
├── menu_guide_neutralization.py   # RDF extraction and transformation pipeline
├── menu_guide_chunking_prototype.py  # Custom RAG with FAISS and OpenAI
├── menu_guide_rag_w_langchain.py     # Production RAG using LangChain
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Access to the MeNu GUIDE turtle file (`menu_guide.ttl`)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd menu-guide-extraction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export MENU_GUIDE_PATH="/path/to/menu_guide.ttl"
   ```

## Usage

### Step 1: Extract Relationships from Knowledge Graph

The extraction script processes the MeNu GUIDE turtle file and generates human-readable sentences about food-compound-disease relationships.

```python
# In menu_guide_neutralization.py, set the path to your turtle file:
# file_path = os.environ.get("MENU_GUIDE_PATH", "/path/to/menu_guide.ttl")

python menu_guide_neutralization.py
```

**Note**: Loading the full knowledge graph (25M+ triples) can take 30+ minutes.

**Output**: `final_sentences.txt` - Contains extracted relationships as natural language sentences.

### Step 2: Build RAG Index

#### Option A: Custom FAISS Implementation

```python
python menu_guide_chunking_prototype.py
```

**Outputs**:
- `rag_chunks.jsonl` - Chunked documents with metadata
- `rag_faiss.index` - FAISS vector index
- `doc_lookup.pkl` - Document metadata lookup

#### Option B: LangChain Implementation (Recommended)

```python
python menu_guide_rag_w_langchain.py
```

**Output**: `rag_faiss_langchain/` - LangChain vector store

### Step 3: Query the System

```python
from menu_guide_chunking_prototype import answer_query

response = answer_query("Which foods are associated with lung cancer?")
print(response)
```

Or with LangChain:

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load saved vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("rag_faiss_langchain", embedding_model)

# Create QA chain
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain("Which foods are associated with lung cancer?")
print(response["result"])
```

## Pipeline Overview

```
┌─────────────────────┐
│  MeNu GUIDE KG      │
│  (menu_guide.ttl)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Extraction &       │
│  Enrichment         │
│  (neutralization.py)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  final_sentences.txt│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Chunking &         │
│  Embedding          │
│  (chunking.py)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FAISS Index +      │
│  LangChain QA       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Natural Language   │
│  Query Interface    │
└─────────────────────┘
```

## Configuration

Key parameters can be adjusted:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `MENU_GUIDE_PATH` | Environment | Path to the turtle file |
| `OPENAI_API_KEY` | Environment | OpenAI API key for LLM |
| `top_k` | `retrieve()` | Number of documents to retrieve (default: 5) |
| `lambda_mult` | LangChain retriever | MMR diversity parameter (0-1, higher = more relevant) |

## Example Queries

- "Which foods are associated with lung cancer?"
- "What is the relationship between creatinine and kidney disease?"
- "Which compounds are biomarkers for diabetes?"
- "What foods contain high levels of specific metabolites?"

## Future Improvements

- [ ] Add evaluation metrics (RAGAS framework for faithfulness and context precision)
- [ ] Implement batch processing for multiple compounds/diseases
- [ ] Add support for additional LLM providers
- [ ] Create CLI interface for end-users
- [ ] Add caching/checkpointing for faster processing

## License

This project is provided for research and educational purposes.

## Acknowledgments

- [MeNu GUIDE](https://github.com/LipiTUM/MeNuGUIDE) - The underlying knowledge graph
- [LangChain](https://langchain.com/) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
