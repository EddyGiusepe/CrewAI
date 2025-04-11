#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Datacamp ---> https://www.datacamp.com/tutorial/crew-ai

Run
===
uv run 1_building_a_web_search_tool_with_crewAI.py
"""
# 1. Raspar um site:
import os
from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
from crewai import Agent, Task, Crew, Process
from crewai_tools import TXTSearchTool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

# Initialize the tool, potentially passing the session
tool = ScrapeWebsiteTool(
    website_url="https://en.wikipedia.org/wiki/Artificial_intelligence"
)

# Extract the text:
text = tool.run()
# print(text)

# 2. Grave o texto extraído em um arquivo:
# Certifique-se de que o diretório existe
os.makedirs("../data", exist_ok=True)

# Initialize the tool
file_writer_tool = FileWriterTool()

# Write content to a file in a specified directory
result = file_writer_tool.run(filename="../data/ai.txt", content=text, overwrite="true")
# print(result)

# 3. Configurar a ferramenta de pesquisa de texto:

# Initialize the tool with a specific text file, so the agent can search within the given text file's content
tool = TXTSearchTool(txt="../data/ai.txt")

# 4. Crie um agente para a tarefa e execute-o:
context = tool.run("O que é Natural Language Processing (NLP)?")

data_analyst = Agent(
    role="Educador",
    goal=f"""Baseado no contexto fornecido, responda a pergunta - O que é Natural Language Processing (NLP)? Contexto - {context}.
          """,
    backstory="""Você é um especialista em dados.
                 Ademais, você sempre deve responder em português brasileiro (pt-br).
              """,
    verbose=False,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.0),  # o3-mini
    tools=[tool],
)

test_task = Task(
    description="""Entenda o assunto e forneça uma resposta correta e factual.
                  Você sempre deve responder em português brasileiro (pt-br).
                  """,
    tools=[tool],
    agent=data_analyst,
    expected_output="Forneça uma resposta correta e factual",
)

crew = Crew(agents=[data_analyst], tasks=[test_task])

output = crew.kickoff()
print(output)
