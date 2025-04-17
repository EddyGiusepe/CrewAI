#!/usr/bin/env python
import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai import LLM
from crewai.flow.flow import Flow, listen, start
from guide_creator_flow.crews.content_crew.content_crew import ContentCrew
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Cria o diretório de saída se não existir:
os.makedirs("output", exist_ok=True)

# Definimos nossos modelos para dados estruturados:
class Section(BaseModel):
    title: str = Field(description="Título da seção")
    description: str = Field(description="Descrição breve do que a seção deve cobrir")

class GuideOutline(BaseModel):
    title: str = Field(description="Título do guia")
    introduction: str = Field(description="Introdução ao tópico")
    target_audience: str = Field(description="Descrição do público-alvo")
    sections: List[Section] = Field(description="Lista de seções no guia")
    conclusion: str = Field(description="Conclusão ou resumo do guia")

# Definimos o estado do fluxo:
class GuideCreatorState(BaseModel):
    topic: str = ""
    audience_level: str = ""
    guide_outline: GuideOutline = None
    sections_content: Dict[str, str] = {}

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Fluxo para criar um guia completo sobre qualquer tópico"""

    @start()
    def get_user_input(self):
        """Obter entrada do usuário sobre o tópico do guia e o público-alvo"""
        print("\n=== Crie seu guia completo ===\n")

        # Obter entrada do usuário:
        self.state.topic = input("Sobre qual tópico você gostaria de criar um guia? ")

        # Obtenha o nível de público-alvo com validação:
        while True:
            audience = input("Quem é seu público-alvo? (iníciante/intermediário/avançado) ").lower()
            if audience in ["iníciante", "intermediário", "avançado"]:
                self.state.audience_level = audience
                break
            print("Por favor, insira 'iníciante', 'intermediário' ou 'avançado'")

        print(f"\nCriando um guia sobre {self.state.topic} para o público {self.state.audience_level}...\n")
        return self.state

    @listen(get_user_input)
    def create_guide_outline(self, state):
        """Cria o outline do guia usando uma chamada direta ao LLM"""
        print("Criando o outline do guia...")

        # Inicializa o LLM:
        llm = LLM(model="openai/gpt-4o-mini", response_format=GuideOutline)

        # Cria as mensagens para o outline:
        messages = [
            {"role": "system", "content": "Você é um assistente útil projetado para produzir JSON."},
            {"role": "user", "content": f"""
            Crie um outline detalhado para um guia completo sobre "{state.topic}" para {state.audience_level} nível de aprendiz.

            O outline deve incluir:
            1. Um título atraente para o guia
            2. Uma introdução ao tópico
            3. 4-6 seções principais que abordam os aspectos mais importantes do tópico
            4. Uma conclusão ou resumo

            Para cada seção, forneça um título claro e uma descrição breve do que deve cobrir.
            """}
        ]

        # Faz a chamada ao LLM com o formato de resposta JSON:
        response = llm.call(messages=messages)

        # Analisa a resposta JSON:
        outline_dict = json.loads(response)
        self.state.guide_outline = GuideOutline(**outline_dict)

        # Salva o outline em um arquivo:
        with open("output/guide_outline.json", "w") as f:
            json.dump(outline_dict, f, indent=2)

        print(f"Guia outline criado com {len(self.state.guide_outline.sections)} seções")
        return self.state.guide_outline

    @listen(create_guide_outline)
    def write_and_compile_guide(self, outline):
        """Escreva todas as seções e compile o guia"""
        print("Escrevendo seções do guia e compilando...")
        completed_sections = []

        # Processa seções uma por uma para manter o fluxo de contexto:
        for section in outline.sections:
            print(f"Processando seção: {section.title}")

            # Constrói o contexto a partir das seções anteriores:
            previous_sections_text = ""
            if completed_sections:
                previous_sections_text = "# Seções escritas anteriormente\n\n"
                for title in completed_sections:
                    previous_sections_text += f"## {title}\n\n"
                    previous_sections_text += self.state.sections_content.get(title, "") + "\n\n"
            else:
                previous_sections_text = "Nenhuma seção escrita ainda."

            # Executa a tripulação de conteúdo para esta seção:
            result = ContentCrew().crew().kickoff(inputs={
                "section_title": section.title,
                "section_description": section.description,
                "audience_level": self.state.audience_level,
                "previous_sections": previous_sections_text,
                "draft_content": ""
            })

            # Armazena o conteúdo:
            self.state.sections_content[section.title] = result.raw
            completed_sections.append(section.title)
            print(f"Seção concluída: {section.title}")

        # Compile o guia final:
        guide_content = f"# {outline.title}\n\n"
        guide_content += f"## Introdução\n\n{outline.introduction}\n\n"

        # Adiciona cada seção na ordem:
        for section in outline.sections:
            section_content = self.state.sections_content.get(section.title, "")
            guide_content += f"\n\n{section_content}\n\n"

        # Adiciona a conclusão:
        guide_content += f"## Conclusão\n\n{outline.conclusion}\n\n"

        # Salva o guia:
        with open("output/complete_guide.md", "w") as f:
            f.write(guide_content)

        print("\nGuia completo compilado e salvo em output/complete_guide.md")
        return "Criação de guia concluída com sucesso"

def kickoff():
    """Executa o fluxo de criação de guia"""
    GuideCreatorFlow().kickoff()
    print("\n=== Fluxo concluído ===")
    print("Seu guia completo está pronto no diretório output.")
    print("Abra output/complete_guide.md para vê-lo.")

def plot():
    """Gera uma visualização do fluxo"""
    flow = GuideCreatorFlow()
    flow.plot("guide_creator_flow")
    print("Visualização do fluxo salva em guide_creator_flow.html")

if __name__ == "__main__":
    kickoff()