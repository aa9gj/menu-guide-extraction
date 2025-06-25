# I used final_sentences.txt for the AI hub assistant. 
# Now I want to turn this into .json file to create my own RAG

import uuid
import json
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import pickle


# chunk final sentences
rag_chunks = []

with open("final_sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        sentence = line.strip()
        if not sentence:
            continue

        # Try to extract title from compound + disease
        try:
            compound = sentence.split("contains")[1].split(",")[0].strip().capitalize()
            disease = sentence.split("biomarker in")[1].split(".")[0].strip().capitalize()
            title = f"{compound} in {disease}"
        except:
            title = "Food-compound-disease relationship"

        rag_chunks.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "body": sentence
        })

# save chunks
with open("rag_chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in rag_chunks:
        f.write(json.dumps(chunk) + "\n")

# Read in chunks
rag_chunks = []

with open("rag_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        rag_chunks.append(doc)

# let's start chunking with something fast and lightweight (good for prototyping)
# this takes a bit of time hence we should think about saving it as pickle 

model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast & lightweight for prototyping but we can always bring in model form huggingface (approved at colgate)
corpus = [doc["body"] for doc in rag_chunks]
embeddings = model.encode(corpus, show_progress_bar=True)

# store in faiss
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# Save index (optional)
faiss.write_index(index, "rag_faiss.index")

# you can also save metadata separately 
# Index to doc metadata (used for retrieval)
doc_lookup = {i: rag_chunks[i] for i in range(len(rag_chunks))}

# Now create the retriever 
def retrieve(query, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    return [doc_lookup[i] for i in indices[0]]

# One of the main ideas to explore is RAGAS framework (faithfullness and context precision (more research))
# Let's build the prompter
def build_prompt(query, retrieved_docs):
    context = "\n\n".join([doc["body"] for doc in retrieved_docs])
    return f"""Use the following biomedical information to answer the question below.

Context:
{context}

Question: {query}
Answer:"""

# Let's answer it with GPT4 but I don't have an API key so moving everything to AI hub with final sentences 
def answer_query(query):
    docs = retrieve(query)
    prompt = build_prompt(query, docs)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
print(answer_query("Which foods are associated with lung cancer?"))

# save everything for later using pickle so you don't have to rebuild the index everytime 
with open("doc_lookup.pkl", "wb") as f:
    pickle.dump(doc_lookup, f)
# to load it again 
with open("doc_lookup.pkl", "wb") as f:
    pickle.dump(doc_lookup, f)

