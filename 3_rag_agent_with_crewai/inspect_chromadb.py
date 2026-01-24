#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

inspect_chromadb.py
===================
Este script é para inspecionar o banco de dados ChromaDB
do CrewAI

Run
===
uv run inspect_chromadb.py
"""
import json
import sqlite3
import sys
from pathlib import Path

# Determina o caminho do ChromaDB baseado no diretório atual:
project_name = Path.cwd().name
db_path = Path.home() / ".local" / "share" / project_name / "chroma.sqlite3"

if not db_path.exists():
    print(f"❌ Banco de dados não encontrado em: {db_path}")
    print("\n💡 Dica: Execute este script do diretório: 3_rag_agent_with_crewai/")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print(f"📁 Localização: {db_path}")
print(f"💾 Tamanho: {db_path.stat().st_size / 1024:.1f} KB")
print("=" * 80)

print("\n📚 COLEÇÕES ARMAZENADAS:")
print("=" * 25)
cursor.execute("SELECT id, name, dimension, config_json_str FROM collections;")
collections = cursor.fetchall()

for i, (col_id, col_name, dimension, config_str) in enumerate(collections, 1):
    print(f'\n{i}. 📦 COLEÇÃO: "{col_name}"')
    print(f"   └─ ID: {col_id}")
    print(f"   └─ Dimensão: {dimension}")

    config = json.loads(config_str)
    if "embedding_function" in config:
        ef = config["embedding_function"]
        if ef.get("type") == "known":
            model_name = ef.get("config", {}).get("model_name", "N/A")
            print(f'   └─ Modelo: {ef.get("name")} ({model_name})')

    # Contar embeddings:
    cursor.execute(
        """
        SELECT COUNT(*) FROM embeddings
        WHERE segment_id IN (SELECT id FROM segments WHERE collection = ?)
    """,
        (col_id,),
    )
    count = cursor.fetchone()[0]
    print(f"   └─ Chunks armazenados: {count}")

print("\n")
print("✅ Banco de dados funcionando corretamente!")

conn.close()
