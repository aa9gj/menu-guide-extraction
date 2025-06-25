## NOW Let's look at using a more sophisticated approach using LangChain

# Let's do it with langchain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import json

# Load your JSONL
docs = []
with open("rag_chunks.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        docs.append(Document(page_content=data["body"], metadata={"title": data["title"]}))

# Create embeddings and vector stores
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save for later
vectorstore.save_local("rag_faiss_langchain")

# I might consider these sentences partially redundant, so maybe I should be using mmr 
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "lambda_mult": 0.7}) # higher value equals more relevance, lower value equals more diversity

# query with openai or any llm
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

# Example query
response = qa_chain("Which foods are associated with lung cancer?")
print(response["result"])
