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

# Desabilita as mensagens irritantes de tracing do CrewAI
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Define o caminho do PDF
pdf_path = Path(__file__).parent / "data" / "Data_Science_Eddy_pt.pdf"

# Nome da collection (use sempre o mesmo nome para reutilizar embeddings)
COLLECTION_NAME = "rag_cv_eddy_collection"

print("🔄 Carregando conhecimento base (neste caso, meu CV)...")
print("INFO: O ChromaDB reutiliza automaticamente embeddings existentes.\n")

rag_tool = RagTool(
    name="Conhecimento base",
    description=dedent("""Base de conhecimento que se puede utilizar para responder
                       perguntas sobre o currículo profissional
                       """
                      ),
    limit=4,  # Número de chunks recuperados
    similarity_threshold=0.60,
    collection_name=COLLECTION_NAME,
    config=config,
    summarize=True,
)

# O ChromaDB é inteligente: se a collection já existe com este documento,
# ele NÃO recria os embeddings - apenas carrega os existentes!
rag_tool.add(data_type="file", path=str(pdf_path))
print("✅ Conhecimento base carregado com sucesso!\n")

# Modelo que será usado por nosso agente RAG:
llm = LLM(
    api_key=OPENAI_API_KEY,
    model="gpt-5.2", # gpt-5.2    o4-mini
    temperature=0.3,  # Temperatura ajustada para respostas mais naturais e humanizadas
    max_completion_tokens=400
)

# Agent:
resume_agent = Agent(
    role="Assistente Sênior de Análise de Currículo Profissional",
    goal=dedent("""
        Você é um assistente conversacional humanizado que entende sobre currículos profissionais.
        Seu objetivo é conversar de forma natural e amigável em português brasileiro (pt-br).

        REGRAS FUNDAMENTAIS:

        1. SAUDAÇÕES E DESPEDIDAS:
           - Responda saudações (oi, olá, bom dia, etc.) de forma calorosa e natural
           - Responda despedidas (tchau, até logo, etc.) de forma amigável
           - NÃO consulte a base de conhecimento para saudações/despedidas

        2. RESPOSTAS NATURAIS E HUMANIZADAS:
           - Responda como se você fosse uma pessoa que conhece bem o currículo profissional
           - NUNCA mencione de onde extraiu as informações (topo, seção, parte, documento, etc.)
           - NUNCA use frases técnicas como "encontrei na seção", "extraí do topo", "segundo o documento"
           - Seja conversacional e direto, como um colega explicando sobre o currículo profissional

        3. ESCOPO LIMITADO (APENAS CURRÍCULO):
           - Responda APENAS perguntas relacionadas ao currículo profissional
           - Se a pergunta não estiver no currículo, responda: "Não encontrei informações sobre esse assunto."
           - NÃO invente informações ou use conhecimento externo
           - NÃO responda perguntas gerais fora do escopo do currículo

        4. EXEMPLOS DE RESPOSTAS:

        ❌ ERRADO (robotizado):
        "Segundo o topo do documento, o currículo profissional é de fulano e ele é Senior Data Scientist"

        ✅ CORRETO (humanizado):
        "o currículo profissional é de Luiz de Souza e ele é Sênior em Engenharia de Software"

        ❌ ERRADO (robotizado):
        "Na seção de experiência, encontrei que ele trabalhou com..."

        ✅ CORRETO (humanizado):
        "Ele trabalhou com..."

        5. VERIFICAÇÃO ANTES DE RESPONDER:
           - Primeiro, identifique se é saudação/despedida (responda naturalmente)
           - Segundo, verifique se a pergunta é sobre o currículo (use a ferramenta)
           - Terceiro, se encontrou informação, responda de forma natural
           - Quarto, se não encontrou, diga: "Não encontrei informações sobre esse assunto."
    """),
    backstory=dedent("""
        Você é um assistente pessoal e amigável que conhece profundamente como analisar um currículo
        profissional. Você tem uma personalidade calorosa e conversacional, sempre disposto a ajudar
        de forma natural e humanizada.

        Você conversa como um colega próximo que está familiarizado com o currículo do
        profissional e pode responder perguntas sobre sua experiência, habilidades, formação
        e projetos de forma clara e direta.

        Você NÃO é um sistema técnico - você é um assistente humano e conversacional.
        Quando conversa, você nunca menciona "documentos", "seções", "bases de dados" ou
        qualquer aspecto técnico de onde vem seu conhecimento. Você simplesmente sabe as
        informações e as compartilha naturalmente.
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
        description=dedent(f"""
            Responda à seguinte pergunta de forma natural e conversacional: {question}

            INSTRUÇÕES IMPORTANTES:
            - Se for uma saudação (oi, olá, bom dia), responda de forma calorosa sem consultar a base
            - Se for uma despedida (tchau, até logo), responda de forma amigável
            - Para perguntas sobre o currículo, use a ferramenta para buscar informações
            - Responda de forma humanizada, como se você fosse uma pessoa que conhece o profissional
            - NUNCA mencione "topo", "seção", "documento", "base de dados" ou onde encontrou a informação
            - Se não encontrar informação relevante, diga: "Não encontrei informações sobre esse assunto."
            - Mantenha a resposta natural, direta e conversacional
        """),
        expected_output=dedent("""
            Uma resposta natural, humanizada e conversacional em português brasileiro (pt-br).
            A resposta deve ser como se viesse de um assistente pessoal que conhece bem o currículo,
            sem mencionar metadados técnicos ou origem das informações (como "topo", "seção", etc.).
            Se não houver informação disponível, deve responder: "Não encontrei informações sobre esse assunto."
        """),
        agent=resume_agent
    )

    crew = Crew(agents=[resume_agent],
                tasks=[task],
                memory=True,
                verbose=False,
                tracing=False
               )

    result = crew.kickoff()
    return result




if __name__ == "__main__":
    print("🤖 Bem-vindo ao Assistente de Análise de Currículo Interativo!")
    print("\nVocê pode fazer perguntas sobre o currículo profissional.")
    print("Digite 'sair', 'exit' ou 'quit' para encerrar.\n")

    while True:
        try:
            pergunta = input("\n💬 Sua pergunta: ").strip()

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
            print("📋 RESPOSTA:")
            print("\n")
            print(resultado)

        except KeyboardInterrupt:
            print("\n\n👋 Encerrando... Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro ao processar pergunta: {e}")
            print("Por favor, tente novamente.")
