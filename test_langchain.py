# Document loader
from langchain_community.document_loaders import TextLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# LLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

import os
print(os.getenv("OPENAI_API_KEY"))



# Load document
loader = TextLoader("data/sample.txt")
docs = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Create embeddings and FAISS DB
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

print("✅ Vector DB created!")

