# Knowledge-Virtual-Agent

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about documents stored in the workspace.

## Project Structure

.env                  # Environment variables configuration  
RAG.py                # Main RAG implementation  
documents/            # Source documents to query  
  ├── Bangalore_info.pdf  
  ├── google_info.txt  
  ├── khabib_info.txt  
  └── sachin_info.txt  
indexes/              # Vector store indexes  
  ├── chroma/         # Chroma vector database  
  ├── faiss/          # FAISS vector indexes  
  └── pinecone/       # Pinecone vector indexes  

## Features

- Support for multiple document types (PDF, TXT, web pages)
- Multiple chunking strategies for text splitting
- Multiple vector store backends (FAISS, Chroma, Pinecone)
- Integration with Azure OpenAI for embeddings and completions

## Getting Started

1. Configure the .env file with your API keys:

api_key = "your_azure_openai_key"  
endpoint = "https://your_instance.openai.azure.com/"  
api_version = "2024-08-01-preview"  
deployment = "gpt-4o"  
azure_embedding_model = "text-embedding-3-small"  
PINECONE_API_KEY = "your_pinecone_key"  
PINECONE_ENVIRONMENT = "us-east-1"  
INDEX_NAME = "learning"  

2. Run the RAG system:
```
from RAG import RAG, AzureChatOpenAI, AzureOpenAIEmbeddings  
import os  
from dotenv import load_dotenv

load_dotenv()

## Initialize models
llm_model = AzureChatOpenAI(
    openai_api_key=os.getenv('api_key'),
    openai_api_base=os.getenv('endpoint'),
    openai_api_version=os.getenv('api_version'),
    deployment_name=os.getenv('deployment'),
    temperature=0.0
)

embedding_model = AzureOpenAIEmbeddings(
    api_key=os.getenv('api_key'),
    api_version=os.getenv('api_version'),
    azure_endpoint=os.getenv('endpoint'),
    azure_deployment=os.getenv('azure_embedding_model')
)

## Initialize RAG with local document path or webpage URL
rag = RAG(
    path="documents/",
    chunking_type="recursive_text_split",
    chunk_size=1000,
    vectorstore_name="faiss",
    llm_model=llm_model,
    embedding_model=embedding_model
)

## Ask questions
response = rag.ask_question("Who founded Google?")
print(response)
```

## Available Vector Stores

- FAISS: Fast, local vector storage (default)
- Chroma: Local vector database with persistence
- Pinecone: Cloud-based vector database (requires API key)

## Text Chunking Methods

- fixed_text_split: Simple character-based splitting
- recursive_text_split: Hierarchical splitting that respects document structure
- token_text_split: Token-aware splitting for LLM context
- sentence_transformer_text_split: Semantic-aware splitting

## Example Usage

The repository comes with several information documents about Google, Sachin Tendulkar, and Khabib Nurmagomedov that can be queried for facts and information.

Requirements

- Python 3.8+
- langchain
- langchain_community
- langchain_text_splitters
- pinecone
- faiss-cpu or faiss-gpu
- azure-openai
