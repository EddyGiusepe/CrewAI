"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script application.py
=====================
Este script contém a aplicação do agente RAG com CrewAI.
Aqui usamos o ChromaDB para armazenar os documentos e o modelo
de embedding OpenAI para criar os embeddings.
Ademais, usamos ragtool como ferramenta para buscar informações
no currículo profissional e o modelo de LLM OpenAI para responder
as perguntas do usuário.

https://docs.crewai.com/en/tools/ai-ml/ragtool

Run
===
uv run app.py

UI with ReactPy
===============
https://reactpy.dev/docs/index.html#
"""

import os
from pathlib import Path
from textwrap import dedent

from ansi_colors import CYAN, GREEN, MAGENTA, RED, RESET, YELLOW
from config_crewai import config
from crewai import LLM, Agent, Crew, Task
from crewai_tools import RagTool
from dotenv import find_dotenv, load_dotenv
from logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Desabilita as mensagens irritantes de tracing do CrewAI
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Define o caminho do PDF
pdf_path = Path(__file__).parent / "data" / "Data_Science_Eddy_pt.pdf"

# Nome da collection (use sempre o mesmo nome para reutilizar embeddings)
COLLECTION_NAME = "rag_cv_eddy_collection"


def load_rag_tool(
    pdf_path: Path,
    collection_name: str = COLLECTION_NAME,
    limit: int = 6,
    similarity_threshold: float = 0.70,
) -> RagTool:
    """
    Carrega e configura o RagTool com o documento PDF.

    O ChromaDB é inteligente: se a collection já existe com este documento,
    ele NÃO recria os embeddings - apenas carrega os existentes!

    Args:
        pdf_path: Caminho para o arquivo PDF
        collection_name: Nome da collection no ChromaDB
        limit: Número de chunks recuperados
        similarity_threshold: Limiar de similaridade para recuperação

    Returns:
        RagTool configurado e carregado com o documento
    """

    rag_tool = RagTool(
        name="Conhecimento base",
        description=dedent(
            """Base de conhecimento que se deve utilizar para responder
                              perguntas sobre o currículo profissional.
                           """
        ),
        limit=limit,
        similarity_threshold=similarity_threshold,
        collection_name=collection_name,
        config=config,
        summarize=True,
    )
    logger.info(f"{CYAN}🔄 Carregando conhecimento base (neste caso, meu CV)...{RESET}")
    logger.info(
        f"{CYAN}O ChromaDB reutiliza automaticamente embeddings existentes.{RESET}"
    )
    rag_tool.add(data_type="file", path=str(pdf_path))
    logger.info(f"{GREEN}✅ Conhecimento base carregado com sucesso!{RESET}")

    return rag_tool


def create_llm(
    api_key: str,
    model: str = "gpt-5.2",  # gpt-5.1 gpt-5.2   gpt-4o-mini # Foi bom --> gpt-4.1
    temperature: float = 0.3,
    max_completion_tokens: int = 2000,
) -> LLM:
    """
    Cria e configura o modelo LLM para o agente RAG.

    Args:
        api_key: Chave da API OpenAI
        model: Nome do modelo a ser usado
        temperature: Temperatura para respostas mais naturais e humanizadas
        max_completion_tokens: Número máximo de tokens na resposta

    Returns:
        Instância configurada do LLM
    """
    return LLM(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )


def create_resume_agent(llm: LLM, rag_tool: RagTool) -> Agent:
    """
    Cria e configura o agente que irá analisar o currículo.

    Args:
        llm: Instância do LLM configurado
        rag_tool: Instância do RagTool carregado

    Returns:
        Agent configurado para analisar o currículo
    """
    return Agent(
        role="Assistente experto em análise de currículo profissional",
        goal=dedent(
            """
            Você é um assistente conversacional experto em análise de currículo profissional.
            Seu objetivo é responder às perguntas do usuário sobre a análise de currículo profissional
            de forma natural, amigável em português brasileiro (pt-br).

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
               - Se a pergunta conter saudação e pergunta sobre o currículo, responda de forma natural e amigável.

            4. EXEMPLOS DE RESPOSTAS:

            ❌ ERRADO (robotizado):
            "Segundo o topo do documento, o currículo profissional é de fulano e ele é um Engenheiro de Software"

            ✅ CORRETO (humanizado):
            "o currículo profissional é de Luiz de Souza e ele é um Arquiteto de Software"

            ❌ ERRADO (robotizado):
            "Na seção de experiência, encontrei que ele trabalhou com..."

            ✅ CORRETO (humanizado):
            "Ele trabalhou com..."

            5. VERIFICAÇÃO ANTES DE RESPONDER:
               - Primeiro, identifique se é saudação/despedida (responda naturalmente)
               - Segundo, verifique se a pergunta é sobre o currículo (use a ferramenta)
               - Terceiro, se encontrou informação, responda de forma natural
               - Quarto, se não encontrou, diga: "Não encontrei informações sobre esse assunto."
        """
        ),
        backstory=dedent(
            """
            Você é um assistente pessoal e amigável que conhece profundamente como analisar um currículo
            profissional. Você tem uma personalidade calorosa e conversacional, sempre disposto a ajudar
            de forma natural e humanizada.

            Você conversa como um colega próximo que está familiarizado com o currículo do
            profissional e pode responder perguntas sobre sua experiência, habilidades, formação
            e projetos de forma clara e direta.

            Você NÃO é um sistema técnico - você é um assistente humanizado e conversacional.
            Quando conversa, você nunca menciona "documentos", "seções", "bases de dados", "topo", "seção", "documento"
            ou qualquer aspecto técnico de onde vem seu conhecimento. Você simplesmente sabe as
            informações e as compartilha naturalmente.
        """
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_retry_limit=3,
    )


# Inicializa os componentes
rag_tool = load_rag_tool(pdf_path)
llm = create_llm(api_key=OPENAI_API_KEY)
resume_agent = create_resume_agent(llm=llm, rag_tool=rag_tool)


def ask_question(question: str) -> str:
    """Faz uma pergunta ao agente RAG"""
    task = Task(
        description=dedent(
            f"""
            Responda à seguinte pergunta de forma natural e conversacional: {question}

            INSTRUÇÕES IMPORTANTES:
            - Se for uma saudação (oi, olá, bom dia, etc.), responda de forma calorosa sem consultar a base
            - Se for uma despedida (tchau, até logo, etc.), responda de forma amigável
            - Para perguntas sobre o currículo, use a ferramenta para buscar informações
            - Responda de forma humanizada, como se você fosse uma pessoa que conhece o currículo profissional
            - NUNCA mencione "topo", "seção", "documento", "base de dados" ou onde encontrou a informação
            - Se não encontrar informação relevante, diga: "Não encontrei informações sobre esse assunto no currículo
              profissional."
            - Mantenha a resposta natural, direta e conversacional
            - Se a pergunta conter saudação e pergunta sobre o currículo, responda de forma natural e amigável.
        """
        ),
        expected_output=dedent(
            """
            Uma resposta natural, humanizada e conversacional em português brasileiro (pt-br).
            A resposta deve ser como se viesse de um assistente experto e que conhece bem o currículo,
            sem mencionar metadados técnicos ou origem das informações (como "topo", "seção", etc.).
            Se não houver informação disponível, deve responder: "Não encontrei informações sobre esse assunto."
        """
        ),
        agent=resume_agent,
    )

    crew = Crew(
        agents=[resume_agent],
        tasks=[task],
        memory=True,  # Por default no crewai text-embedding-3-small, enables short-term, long-term, and entity memory
        verbose=False,
        tracing=False,
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    logger.info(
        f"{YELLOW}🤖 Bem-vindo ao Assistente de Análise de Currículo Interativo! 🤖{RESET}"
    )
    logger.info(f"{MAGENTA}Digite 'sair', 'exit' ou 'quit' para encerrar.{RESET}")

    while True:
        try:
            pergunta = input(f"{RED}💬 Sua pergunta: {RESET}").strip()

            if pergunta.lower() in ["sair", "exit", "quit", "q"]:
                logger.info(
                    f"{GREEN}👋 Obrigado por usar o assistente! Até logo!{RESET}"
                )
                break

            # Verifica se a pergunta não está vazia
            if not pergunta:
                logger.info(f"{RED}⚠️  Por favor, digite uma pergunta válida.{RESET}")
                continue

            # Processa a pergunta
            logger.info(f"{CYAN}🔍 Processando sua pergunta...{RESET}")
            resultado = ask_question(pergunta)
            print(f"{CYAN}📋 RESPOSTA:{RESET}")
            print(resultado)

        except KeyboardInterrupt:
            logger.info(f"{GREEN}👋 Encerrando... Até logo!{RESET}")
            break
        except Exception as e:
            logger.error(f"{RED}❌ Erro ao processar pergunta: {e}{RESET}")
            logger.info(f"{RED}Por favor, tente novamente.{RESET}")
