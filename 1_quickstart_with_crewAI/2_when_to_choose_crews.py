#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

As equipes (crews) são ideais quando
------------------------------------

1. Você precisa de inteligência colaborativa - Vários agentes com diferentes especializações
                                               precisam trabalhar juntos
2. O problema requer pensamento emergente - A solução se beneficia de diferentes perspectivas
                                            e abordagens
3. A tarefa é principalmente criativa ou analítica - O trabalho envolve pesquisa, criação de
                                                     conteúdo ou análise
4. Você valoriza a adaptabilidade em vez de uma estrutura rígida - O fluxo de trabalho pode se
                                                                   beneficiar da autonomia do agente
5. O formato de saída pode ser um tanto flexível - alguma variação na estrutura de saída é aceitável

A seguir, um exemplo de uma equipe (crew) para análise de mercado
-----------------------------------------------------------------
"""
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

# Criar agentes especializados:
researcher = Agent(
    role="Especialista em Pesquisa de Mercado",
    goal="Encontrar dados de mercado detalhados sobre tecnologias emergentes",
    backstory="Você é um especialista em descobrir tendências de mercado e coletar dados.",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
)

analyst = Agent(
    role="Analista de Mercado",
    goal="Analisar dados de mercado e identificar oportunidades chaves",
    backstory="Você é um especialista em interpretar dados de mercado e identificar insights valiosos.",
    llm=ChatOpenAI(model="o3-mini", temperature=0.0),  # o3-mini   gpt-4o-mini
)

# Definir suas tarefas:
research_task = Task(
    description="Pesquise o cenário de mercado atual para soluções de saúde com IA",
    expected_output="Dados de mercado detalhados, incluindo principais players, tamanho do mercado e tendências de crescimento",
    agent=researcher,
)

analysis_task = Task(
    description="Analisar os dados de mercado e identificar as 3 melhores oportunidades de investimento",
    expected_output="Relatório de análise com 3 oportunidades de investimento recomendadas e razões",
    agent=analyst,
    context=[research_task],
)

# Criar a equipe:
market_analysis_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
)

# Executar a equipe:
result = market_analysis_crew.kickoff()

print(result)
