#!/usr/bin/env python
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Run (onde está o main.py)
=========================
uv run main.py

ou na raíz do projeto (research_crew)
=====================================
crewai run
"""
import os
from research_crew.crew import ResearchCrew
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


# Cria o diretório de saída se não existir:
os.makedirs("output", exist_ok=True)


def run():
    """
    Executa a equipe de pesquisa.
    """
    inputs = {"topic": "Inteligência Artificial na Saúde"}

    # Cria e executa a equipe:
    result = ResearchCrew().crew().kickoff(inputs=inputs)

    # Imprime o resultado:
    print("\n\n=== RELATÓRIO FINAL ===\n\n")
    print(result.raw)

    print("\n\nO relatório foi salvo em output/report.md")


if __name__ == "__main__":
    run()
