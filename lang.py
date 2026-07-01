import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()

# 1. Load Data

file_path = "data.txt"  

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# 2. Split into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# 3. Embeddings 

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Vector DB

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. LLM

llm = ChatGroq(
    model="llama-3.1-8b-instant",
   
    groq_api_key = os.getenv("GROQ_API_KEY") )

# 6. Prompt

prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant.

Use ONLY the provided context to answer.
If the answer is not in context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""")

# 7. RAG Chain

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n LangChain RAG system ready! Type 'exit' to quit\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    answer = rag_chain.invoke(query)

    print("\n Answer:\n", answer, "\n")
