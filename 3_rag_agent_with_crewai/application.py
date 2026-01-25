"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script application.py
=====================
This script contains the application of the RAG agent with CrewAI.
Here we use ChromaDB to store the documents and the OpenAI embedding
model to create the embeddings. Moreover, we use ragtool as a tool
to search for information in the professional curriculum and the
OpenAI LLM model to answer the user's questions.

https://docs.crewai.com/en/tools/ai-ml/ragtool

Run
===
uv run application.py

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

# Disable crewAI tracing messages:
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Define the path to the PDF file:
pdf_path = Path(__file__).parent / "data" / "Data_Science_Eddy_pt.pdf"

# Name of the collection (use always the same name to reuse embeddings):
COLLECTION_NAME = "rag_cv_eddy_collection"


def load_rag_tool(
    pdf_path: Path,
    collection_name: str = COLLECTION_NAME,
    limit: int = 6,
    similarity_threshold: float = 0.70,
) -> RagTool:
    """
    Loads and configures the RagTool with the PDF file.

    ChromaDB is intelligent: if the collection already exists with this document,
    it does NOT recreate the embeddings - it only loads the existing ones!

    Args:
        pdf_path: Path to the PDF file
        collection_name: Name of the collection in ChromaDB
        limit: Number of chunks retrieved
        similarity_threshold: Similarity threshold for retrieval

    Returns:
        RagTool configured and loaded with the PDF file
    """

    rag_tool = RagTool(
        name="Knowledge base",
        description=dedent(
            """Knowledge base to be used to answer questions about the
                              professional curriculum.
                           """
        ),
        limit=limit,
        similarity_threshold=similarity_threshold,
        collection_name=collection_name,
        config=config,
        summarize=True,
    )
    logger.info(f"{CYAN}🔄 Loading knowledge base (in this case, my CV)...{RESET}")
    logger.info(f"{CYAN}ChromaDB automatically reuses existing embeddings.{RESET}")
    rag_tool.add(data_type="file", path=str(pdf_path))
    logger.info(f"{GREEN}✅ Knowledge base loaded successfully!{RESET}")

    return rag_tool


def create_llm(
    api_key: str,
    model: str = "gpt-5.2",  # gpt-5.1 gpt-5.2   gpt-4o-mini # Foi bom --> gpt-4.1
    temperature: float = 0.3,
    max_completion_tokens: int = 2000,
) -> LLM:
    """
    Creates and configures the LLM model for the RAG agent.

    Args:
        api_key: OpenAI API key
        model: Name of the model to be used
        temperature: Temperature for more natural and humanized responses
        max_completion_tokens: Maximum number of tokens in the response

    Returns:
        Configured instance of the LLM
    """
    return LLM(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )


def create_resume_agent(llm: LLM, rag_tool: RagTool) -> Agent:
    """
    Creates and configures the agent that will analyze the curriculum.

    Args:
        llm: Configured instance of the LLM
        rag_tool: Configured instance of the RagTool

    Returns:
        Agent configured to analyze the curriculum
    """
    return Agent(
        role="Expert assistant in professional curriculum analysis",
        goal=dedent(
            """
            You are an expert conversational assistant in professional curriculum analysis.
            Your objective is to answer the user's questions about the professional curriculum
            analysis in a natural, friendly way in English.

            FUNDAMENTAL RULES:

            1. GREETINGS AND FAREWELLS:
               - Respond to greetings (hi, hello, good day, etc.) in a warm and natural way
               - Respond to farewells (bye, goodbye, etc.) in a friendly way
               - DO NOT consult the knowledge base for greetings/farewells

            2. NATURAL AND HUMANIZED RESPONSES:
               - Respond as if you were a person who knows the professional curriculum well
               - DO NOT mention where you found the information (top, section, part, document, etc.)
               - DO NOT use technical phrases like "found in the section", "extracted from the top",
                 "according to the document", etc.
               - Be conversational and direct, like a colleague explaining about the professional curriculum

            3. LIMITED SCOPE (ONLY CURRICULUM):
               - Respond only questions related to the professional curriculum
               - If the question is not related to the curriculum, respond: "I did not find information about
                 that subject."
               - DO NOT invent information or use external knowledge
               - DO NOT answer general questions outside the scope of the curriculum
               - If the question contains a greeting and asks about the curriculum, respond in a natural and
                 friendly way.

            4. EXAMPLES OF RESPONSES:

            ❌ WRONG (robotized):
            "According to the top of the document, the professional curriculum is of fulano and he
             is a Software Engineer"

            ✅ RIGHT (humanized):
            "The professional curriculum is of Luiz de Souza and he is a Software Architect"

            ❌ WRONG (robotized):
            "In the experience section, I found that he worked with..."

            ✅ RIGHT (humanized):
            "He worked with..."

            5. VERIFICATION BEFORE RESPONDING:
               - First, identify if it is a greeting/farewell (respond naturally)
               - Second, check if the question is about the curriculum (use the tool)
               - Third, if you found information, respond in a natural way
               - Fourth, if you did not find information, say: "I did not find information about that subject."
        """
        ),
        backstory=dedent(
            """
            You are a personal and friendly assistant who deeply knows how to analyze a professional curriculum.
            You have a warm and conversational personality, always willing to help in a natural and humanized way.

            You converse like a close colleague who is familiar with the professional curriculum and can answer
            questions about their experience, skills, education, and projects in a clear and direct way.

            You are NOT a technical system - you are a humanized and conversational assistant.
            When you converse, you never mention "documents", "sections", "databases", "top", "section", "document"
            or any technical aspect of where your knowledge comes from. You simply know the information and share it
            naturally.
        """
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_retry_limit=3,
    )


# Initialize the components:
rag_tool = load_rag_tool(pdf_path)
llm = create_llm(api_key=OPENAI_API_KEY)
resume_agent = create_resume_agent(llm=llm, rag_tool=rag_tool)


def ask_question(question: str) -> str:
    """Ask a question to the RAG agent"""
    task = Task(
        description=dedent(
            f"""
            Respond to the following question in a natural and conversational way: {question}

            IMPORTANT INSTRUCTIONS:
            - If it is a greeting (hi, hello, good day, etc.), respond in a warm way without consulting the base
            - If it is a farewell (bye, goodbye, etc.), respond in a friendly way
            - For questions about the curriculum, use the tool to search for information
            - Respond in a humanized way, as if you were a person who knows the professional curriculum
            - DO NOT mention "top", "section", "document", "database" or where you found the information
            - If you do not find relevant information, say: "I did not find information about that subject in the
              professional curriculum."
            - Keep the response natural, direct and conversational
            - If the question contains a greeting and asks about the curriculum, respond in a natural and friendly way.
        """
        ),
        expected_output=dedent(
            """
            A natural, humanized and conversational response in Portuguese (pt-br).
            The response should be like if it came from an expert assistant who knows the curriculum well,
            without mentioning technical metadata or the origin of the information (like "top", "section", etc.).
            If there is no information available, respond: "I did not find information about that subject in the
            professional curriculum."
        """
        ),
        agent=resume_agent,
    )

    crew = Crew(
        agents=[resume_agent],
        tasks=[task],
        memory=True,  # By default in crewai text-embedding-3-small, enables short-term, long-term, and entity memory
        verbose=False,
        tracing=False,
    )

    result = crew.kickoff()
    return result


if __name__ == "__main__":
    logger.info(
        f"{YELLOW}🤖 Welcome to the RAG Interactive Resume Analysis Agent! 🤖{RESET}"
    )
    logger.info(f"{MAGENTA}Type 'exit', 'quit' or 'q' to end.{RESET}")

    while True:
        try:
            question = input(f"{RED}💬 Your question: {RESET}").strip()

            if question.lower() in ["exit", "quit", "q"]:
                logger.info(
                    f"{GREEN}👋 Thank you for using the assistant! Goodbye!{RESET}"
                )
                break

            # Check if the question is not empty:
            if not question:
                logger.info(f"{RED}⚠️  Please enter a valid question.{RESET}")
                continue

            # Process the question:
            logger.info(f"{CYAN}🔍 Processing your question...{RESET}")
            result = ask_question(question)
            print(f"{CYAN}📋 ANSWER:{RESET}")
            print(result)

        except KeyboardInterrupt:
            logger.info(f"{GREEN}👋 Ending... Goodbye!{RESET}")
            break
        except Exception as e:
            logger.error(f"{RED}❌ Error processing question: {e}{RESET}")
            logger.info(f"{RED}Please try again.{RESET}")
