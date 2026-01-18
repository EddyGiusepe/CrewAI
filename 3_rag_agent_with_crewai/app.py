"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Agente RAG Interativo com Histórico - CrewAI
=============================================
Versão avançada com histórico de conversação e salvamento automático.


https://docs.crewai.com/en/tools/ai-ml/ragtool

Run
===
uv run app.py
"""
import os
from pathlib import Path
from textwrap import dedent

from config import config
from crewai import LLM, Agent, Crew, Task
from crewai_tools import RagTool
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define o caminho do PDF
pdf_path = Path(__file__).parent / "data" / "Data_Science_Eddy_pt.pdf"

print("🔄 Carregando conhecimento base (neste caso, meu CV)...")
rag_tool = RagTool(
    name="Conhecimento base",
    description="Base de conhecimento que se puede utilizar para responder perguntas sobre o currículo profissional",
    limit=4, # Número de chunks recuperados
    similarity_threshold=0.60,
    collection_name="rag_cv_eddy_collection",
    config=config,
    summarize=True,
)

rag_tool.add(data_type="file", path=str(pdf_path))
print("✅ Conhecimento base carregado com sucesso!\n")

# Modelo que será usado por nosso agente RAG:
llm = LLM(
    api_key=OPENAI_API_KEY,
    model="gpt-5.2", # gpt-5.2    o4-mini
    temperature=0.0,
    max_completion_tokens=400
)

# Agent:
resume_agent = Agent(
    role="Assistente Sênior de Análise de Currículo Profissional",
    goal=dedent("""
        Responder perguntas de forma concisa, clara, factual e precisa baseada na base de conhecimento
        fornecida sobre o currículo profissional. Ademais, você deve responder em português
        brasileiro (pt-br). SEMPRE responda perguntas baseadas no conhecimento fornecido e
        se a pergunta naõ for baseada no conhecimento fornecido, responda: "Não tenho informações
        sobre esse assunto."
        Também, responda saudações e despedidas apropriadas de forma natural e humanizada.
    """),
    backstory=dedent("""
        Você é um especialista em análise de currículos profissionais
        com anos de experiência em recrutamento técnico. Você analisa
        currículos de forma objetiva e detalhada, especializando-se em
        sistemas Agênticos RAG para fornecer respostas precisas.
    """),
    verbose=False,
    allow_delegation=False,
    llm=llm,
    tools=[rag_tool],
    max_retry_limit=3
)


def ask_question(question: str) -> str:
    """Faz uma pergunta ao agente RAG"""
    task = Task(
        description=f"Responda à seguinte pergunta sobre o currículo profissional: {question}",
        expected_output="""Uma resposta detalhada, factual e precisa baseada da base de conhecimento sobre
                         o currículo profissional""",
        agent=resume_agent
    )

    crew = Crew(agents=[resume_agent],
                tasks=[task],
                verbose=False,
                tracing=False
               )

    result = crew.kickoff()
    return result




if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 Bem-vindo ao Assistente de Análise de Currículo Interativo!")
    print("="*70)
    print("\nVocê pode fazer perguntas sobre o currículo profissional.")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.\n")

    while True:
        try:
            # Solicita a pergunta do usuário
            pergunta = input("\n💬 Sua pergunta: ").strip()

            # Verifica se o usuário quer sair
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Obrigado por usar o assistente! Até logo!")
                break

            # Verifica se a pergunta não está vazia
            if not pergunta:
                print("⚠️  Por favor, digite uma pergunta válida.")
                continue

            # Processa a pergunta
            print("\n🔍 Processando sua pergunta...")
            resultado = ask_question(pergunta)
            print("\n" + "="*70)
            print("📋 RESPOSTA:")
            print("="*70)
            print(resultado)
            print("="*70)

        except KeyboardInterrupt:
            print("\n\n👋 Encerrando... Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro ao processar pergunta: {e}")
            print("Por favor, tente novamente.")
