#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Run
===
streamlit run app.py
"""
from crewai import Agent, Task, Crew
from crewai.tools import tool
import wikipedia
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

@tool("wikipedia_lookup")
def wikipedia_lookup(q: str) -> str:
    """Pesquise uma consulta na Wikipédia e retorne um resumo"""
    return wikipedia.page(q).summary

intelligent_wikipedia = Agent(
        role="Pesquisador",
        goal="Você pesquisa tópicos usando Wikipedia e relata os resultados",
        backstory="Você é um escritor e editor experiente",
        tools=[wikipedia_lookup],
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0) # "gpt-4o"   ou  "gpt-4o-mini"  ou  "gpt-3.5-turbo-0125"
    )

# A função run define a tarefa e executa a crew e retorna o resultado:
def run(query: str):
    task = Task(
        description=query,
        expected_output='Um texto curto baseado nas informações da ferramenta',
        agent=intelligent_wikipedia,
        tools=[wikipedia_lookup]
    )

    crew = Crew(
        agents=[intelligent_wikipedia],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()
    #print(result)
    task_output = task.output
    return task_output.raw

###############################
#        Usando Streamlit     #
###############################
import streamlit as st

# Configuração da barra lateral com informações do autor
st.sidebar.header("Sobre o Autor")
st.sidebar.markdown("""
**Senior Data Scientist:** Dr. Eddy Giusepe Chirinos Isidro   
**E-mail:** eddychirinos.unac@gmail.com  
**LinkedIn:** [linkedin](https://www.linkedin.com/in/eddy-giusepe-chirinos-isidro-phd-85a43a42/)  
**DagsHub:** [DagsHub](https://dagshub.com/EddyGiusepe)
""")

st.markdown("""
    <h1 style="text-align: center;">Intelligent Wikipedia</h1>
    """, unsafe_allow_html=True)

# Campo de entrada para a pergunta:
if query := st.text_input("Digite sua pergunta: "):
    # Função fictícia `run` que processa a pergunta e retorna uma resposta
    answer = run(query)
    st.markdown(answer)
