#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

config_crewai.py
================
Este script configura o RAG Tool para o CrewAI.
Usa o banco de dados ChromaDB para armazenar
os documentos e o modelo de embedding OpenAI
para criar os embeddings.
Run
===
uv run config_crewai.py
"""
import os

from crewai_tools.tools.rag import ProviderSpec, RagToolConfig, VectorDbConfig
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuração do VectorDB (ChromaDB)
vectordb: VectorDbConfig = {
    "provider": "chromadb",
    "config": {
        "collection_name": "rag_cv_eddy_collection",
    },
}

# Configuração do modelo de embedding
embedding_model: ProviderSpec = {
    "provider": "openai",
    "config": {"model_name": "text-embedding-3-large", "api_key": OPENAI_API_KEY},
}

# Configuração completa do RAG Tool
# Nota: RagToolConfig aceita apenas 'vectordb' e 'embedding_model'
# O LLM é configurado separadamente no Agent, não aqui
config: RagToolConfig = {"vectordb": vectordb, "embedding_model": embedding_model}
