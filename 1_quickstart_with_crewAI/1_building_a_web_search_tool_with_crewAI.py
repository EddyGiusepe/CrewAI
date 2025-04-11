#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Datacamp ---> https://www.datacamp.com/tutorial/crew-ai

crewAi --> https://docs.crewai.com/introduction
"""
# 1. Raspar um site:
from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
import requests
import os

# Initialize the tool, potentially passing the session
tool = ScrapeWebsiteTool(website_url='https://en.wikipedia.org/wiki/Artificial_intelligence')  

# Extract the text:
text = tool.run()
print(text)

# 2. Grave o texto extraído em um arquivo:
# Certifique-se de que o diretório existe
os.makedirs("data", exist_ok=True)

# Initialize the tool
file_writer_tool = FileWriterTool()

# Write content to a file in a specified directory
result = file_writer_tool.run(
    filename="./data/ai.txt",
    content=text,
    overwrite="true"
)

print(result)