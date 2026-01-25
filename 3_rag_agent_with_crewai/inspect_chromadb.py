#!/usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

inspect_chromadb.py
===================
This script is to inspect the ChromaDB database
of the CrewAI.

Run
===
uv run inspect_chromadb.py
"""
import json
import sqlite3
import sys
from pathlib import Path

# Determine the path to the ChromaDB database based on the current directory:
project_name = Path.cwd().name
db_path = Path.home() / ".local" / "share" / project_name / "chroma.sqlite3"

if not db_path.exists():
    print(f"❌ ChromaDB database not found in: {db_path}")
    print("\n💡 Tip: Execute this script from the directory: 3_rag_agent_with_crewai/")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print(f"📁 Location: {db_path}")
print(f"💾 Size: {db_path.stat().st_size / 1024:.1f} KB")
print("=" * 80)

print("\n📚 STORED COLLECTIONS:")
print("=" * 25)
cursor.execute("SELECT id, name, dimension, config_json_str FROM collections;")
collections = cursor.fetchall()

for i, (col_id, col_name, dimension, config_str) in enumerate(collections, 1):
    print(f'\n{i}. 📦 COLLECTION: "{col_name}"')
    print(f"   └─ ID: {col_id}")
    print(f"   └─ Dimension: {dimension}")

    config = json.loads(config_str)
    if "embedding_function" in config:
        ef = config["embedding_function"]
        if ef.get("type") == "known":
            model_name = ef.get("config", {}).get("model_name", "N/A")
            print(f'   └─ Model: {ef.get("name")} ({model_name})')

    # Count embeddings:
    cursor.execute(
        """
        SELECT COUNT(*) FROM embeddings
        WHERE segment_id IN (SELECT id FROM segments WHERE collection = ?)
    """,
        (col_id,),
    )
    count = cursor.fetchone()[0]
    print(f"   └─ Stored chunks: {count}")

print("\n")
print("✅ ChromaDB database is working correctly!")

conn.close()
