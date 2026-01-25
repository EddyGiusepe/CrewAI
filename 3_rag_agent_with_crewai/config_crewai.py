#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

config_crewai.py
================
This script configures the RAG Tool for CrewAI.
Uses the ChromaDB database to store the documents and the OpenAI embedding
model to create the embeddings.

Run
===
uv run config_crewai.py
"""
import os

from crewai_tools.tools.rag import ProviderSpec, RagToolConfig, VectorDbConfig
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # Read local .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration of the VectorDB (ChromaDB):
vectordb: VectorDbConfig = {
    "provider": "chromadb",
    "config": {
        "collection_name": "rag_cv_eddy_collection",
    },
}

# Configuration of the embedding model:
embedding_model: ProviderSpec = {
    "provider": "openai",
    "config": {"model_name": "text-embedding-3-large", "api_key": OPENAI_API_KEY},
}

# Complete configuration of the RAG Tool
# Note: RagToolConfig accepts only 'vectordb' and 'embedding_model'
# The LLM is configured separately in the Agent, not here
config: RagToolConfig = {"vectordb": vectordb, "embedding_model": embedding_model}
