"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script ui.py
============
Este script contém a interface de chat estilo ChatGPT
com tema escuro moderno.
Para a criação desta interface, usamos o framework ReactPy.

Run
===
uv run ui.py

Acesse: http://localhost:8000
"""

import asyncio

import uvicorn
from application import ask_question  # Importa o agente RAG do app.py
from fastapi import FastAPI
from reactpy import component, hooks, html
from reactpy.backend.fastapi import configure

# =========================
# ESTILOS CSS (Tema Escuro)
# =========================
COLORS = {
    "background": "#1a1a2e",
    "card": "#16213e",
    "accent": "#0f3460",
    "primary": "#e94560",
    "primary_hover": "#ff6b6b",
    "text": "#eaeaea",
    "text_muted": "#a0a0a0",
    "user_bubble": "#0f3460",
    "assistant_bubble": "#16213e",
    "input_bg": "#0f3460",
    "border": "#2a2a4a",
}

FONTS = {
    "main": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
}


# ===========
# COMPONENTES
# ===========
@component
def chat_message(role: str, content: str):
    """Renderiza uma mensagem individual do chat."""
    is_user = role == "user"

    container_style = {
        "display": "flex",
        "justifyContent": "flex-end" if is_user else "flex-start",
        "marginBottom": "16px",
        "padding": "0 20px",
    }

    bubble_style = {
        "maxWidth": "70%",
        "padding": "14px 18px",
        "borderRadius": "18px",
        "backgroundColor": (
            COLORS["user_bubble"] if is_user else COLORS["assistant_bubble"]
        ),
        "color": COLORS["text"],
        "fontFamily": FONTS["main"],
        "fontSize": "14px",
        "lineHeight": "1.6",
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.3)",
        "border": f"1px solid {COLORS['border']}",
        "borderBottomRightRadius": "4px" if is_user else "18px",
        "borderBottomLeftRadius": "18px" if is_user else "4px",
        "whiteSpace": "pre-wrap",
        "wordBreak": "break-word",
    }

    role_label_style = {
        "fontSize": "11px",
        "color": COLORS["primary"] if is_user else COLORS["text_muted"],
        "marginBottom": "6px",
        "fontWeight": "600",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
    }

    role_text = "Você" if is_user else "🤖 Assistente"

    return html.div(
        {"style": container_style},
        html.div(
            {"style": bubble_style},
            html.div({"style": role_label_style}, role_text),
            html.div(content),
        ),
    )


@component
def loading_indicator():
    """Indicador de carregamento animado."""
    container_style = {
        "display": "flex",
        "justifyContent": "flex-start",
        "marginBottom": "16px",
        "padding": "0 20px",
    }

    bubble_style = {
        "padding": "14px 18px",
        "borderRadius": "18px",
        "backgroundColor": COLORS["assistant_bubble"],
        "color": COLORS["text_muted"],
        "fontFamily": FONTS["main"],
        "fontSize": "14px",
        "border": f"1px solid {COLORS['border']}",
        "borderBottomLeftRadius": "4px",
    }

    return html.div(
        {"style": container_style},
        html.div(
            {"style": bubble_style},
            html.div(
                {
                    "style": {
                        "fontSize": "11px",
                        "color": COLORS["text_muted"],
                        "marginBottom": "6px",
                        "fontWeight": "600",
                        "textTransform": "uppercase",
                    }
                },
                "Assistente",
            ),
            html.div("Processando sua pergunta..."),
        ),
    )


@component
def chat_input(on_send, is_loading: bool):
    """Campo de input para enviar mensagens."""
    input_value, set_input_value = hooks.use_state("")

    container_style = {
        "display": "flex",
        "gap": "12px",
        "padding": "20px",
        "backgroundColor": COLORS["card"],
        "borderTop": f"1px solid {COLORS['border']}",
    }

    input_style = {
        "flex": "1",
        "padding": "14px 18px",
        "borderRadius": "12px",
        "border": f"2px solid {COLORS['border']}",
        "backgroundColor": COLORS["input_bg"],
        "color": COLORS["text"],
        "fontFamily": FONTS["main"],
        "fontSize": "14px",
        "outline": "none",
        "transition": "border-color 0.2s ease",
    }

    button_style = {
        "padding": "14px 28px",
        "borderRadius": "12px",
        "border": "none",
        "backgroundColor": COLORS["primary"] if not is_loading else COLORS["accent"],
        "color": COLORS["text"],
        "fontFamily": FONTS["main"],
        "fontSize": "14px",
        "fontWeight": "600",
        "cursor": "pointer" if not is_loading else "not-allowed",
        "transition": "all 0.2s ease",
        "opacity": "1" if not is_loading else "0.6",
    }

    def handle_input_change(event):
        set_input_value(event["target"]["value"])

    def handle_submit(event):
        if input_value.strip() and not is_loading:
            on_send(input_value.strip())
            set_input_value("")

    def handle_key_press(event):
        if event.get("key") == "Enter" and not event.get("shiftKey"):
            handle_submit(event)

    return html.div(
        {"style": container_style},
        html.input(
            {
                "type": "text",
                "value": input_value,
                "onChange": handle_input_change,
                "onKeyPress": handle_key_press,
                "placeholder": "Digite sua pergunta sobre o currículo...",
                "style": input_style,
                "disabled": is_loading,
            }
        ),
        html.button(
            {
                "onClick": handle_submit,
                "style": button_style,
                "disabled": is_loading,
            },
            "Enviar" if not is_loading else "...",
        ),
    )


@component
def header_ui():
    """Cabeçalho da aplicação."""
    header_style = {
        "padding": "24px 20px",
        "backgroundColor": COLORS["card"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "textAlign": "center",
    }

    title_style = {
        "margin": "0",
        "color": COLORS["text"],
        "fontFamily": FONTS["main"],
        "fontSize": "24px",
        "fontWeight": "700",
        "letterSpacing": "-0.5px",
    }

    # subtitle_style = {
    #    "margin": "8px 0 0 0",
    #    "color": COLORS["text_muted"],
    #    "fontFamily": FONTS["main"],
    #    "fontSize": "13px",
    # }

    accent_style = {
        "color": COLORS["primary"],
    }

    return html.header(
        {"style": header_style},
        html.h1(
            {"style": title_style},
            html.span({"style": accent_style}, "RAG "),
            "Assistente de Currículo",
        ),
        # html.p(
        #    {"style": subtitle_style},
        #    "Faça perguntas sobre o currículo profissional",
        # ),
    )


@component
def welcome_message():
    """Mensagem de boas-vindas quando não há mensagens."""
    container_style = {
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "height": "100%",
        "padding": "40px",
        "textAlign": "center",
    }

    icon_style = {
        "fontSize": "64px",
        "marginBottom": "20px",
    }

    title_style = {
        "color": COLORS["text"],
        "fontFamily": FONTS["main"],
        "fontSize": "20px",
        "fontWeight": "600",
        "margin": "0 0 12px 0",
    }

    text_style = {
        "color": COLORS["text_muted"],
        "fontFamily": FONTS["main"],
        "fontSize": "14px",
        "lineHeight": "1.6",
        "maxWidth": "400px",
    }

    suggestions_style = {
        "marginTop": "24px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "8px",
    }

    suggestion_style = {
        "padding": "10px 16px",
        "backgroundColor": COLORS["accent"],
        "borderRadius": "8px",
        "color": COLORS["text_muted"],
        "fontFamily": FONTS["main"],
        "fontSize": "13px",
        "border": f"1px solid {COLORS['border']}",
    }

    return html.div(
        {"style": container_style},
        html.div({"style": icon_style}, "💼"),
        html.h2({"style": title_style}, "Bem-vindo ao Assistente de Currículo!"),
        html.p(
            {"style": text_style},
            "Estou aqui para ajudá-lo com informações sobre o currículo profissional.",
        ),
        html.div(
            {"style": suggestions_style},
            html.div(
                {"style": suggestion_style},
                "💡 Exemplo 1: Sobre quem está falando o currículo profissional?",
            ),
            html.div(
                {"style": suggestion_style},
                "💡 Exemplo 2: Quais são as habilidades técnicas desse profissional?",
            ),
            html.div(
                {"style": suggestion_style},
                "💡 Exemplo 3: Qual é a formação acadêmica desse profissional?",
            ),
        ),
    )


@component
def chat_app():
    """Componente principal do chat."""
    messages, set_messages = hooks.use_state([])
    is_loading, set_is_loading = hooks.use_state(False)
    pending_question, set_pending_question = hooks.use_state(None)

    @hooks.use_effect(dependencies=[pending_question])
    async def process_pending_question():
        """Processa a pergunta pendente quando ela muda."""
        if pending_question is None:
            return

        try:
            # Executa a função ask_question em uma thread separada
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, ask_question, pending_question)
            response = str(result)
        except Exception as e:
            response = f"Erro ao processar pergunta: {e!s}"

        # Adiciona a resposta do assistente
        assistant_message = {"role": "assistant", "content": response}
        set_messages(lambda prev: [*prev, assistant_message])
        set_is_loading(False)
        set_pending_question(None)

    def handle_send(message: str):
        """Manipula o envio de uma nova mensagem."""
        # Adiciona a mensagem do usuário
        new_user_message = {"role": "user", "content": message}
        set_messages(lambda prev: [*prev, new_user_message])
        set_is_loading(True)
        set_pending_question(message)

    # Estilos do container principal
    app_style = {
        "display": "flex",
        "flexDirection": "column",
        "height": "100vh",
        "backgroundColor": COLORS["background"],
        "fontFamily": FONTS["main"],
    }

    messages_container_style = {
        "flex": "1",
        "overflowY": "auto",
        "padding": "20px 0",
        "scrollBehavior": "smooth",
    }

    # CSS global para scrollbar e fonte
    global_style = f"""
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: {COLORS['background']};
            margin: 0;
            padding: 0;
        }}

        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {COLORS['card']};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {COLORS['accent']};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['primary']};
        }}

        input:focus {{
            border-color: {COLORS['primary']} !important;
        }}

        button:hover:not(:disabled) {{
            background-color: {COLORS['primary_hover']} !important;
            transform: translateY(-1px);
        }}
    """

    # Renderiza as mensagens
    message_elements = []
    if messages:
        for idx, msg in enumerate(messages):
            message_elements.append(
                chat_message(role=msg["role"], content=msg["content"], key=str(idx))
            )

    return html.div(
        {"style": app_style},
        html.style(global_style),
        header_ui(),
        html.div(
            {"style": messages_container_style},
            welcome_message() if not messages else message_elements,
            loading_indicator() if is_loading else None,
        ),
        chat_input(on_send=handle_send, is_loading=is_loading),
    )


# ====================
# EXECUÇÃO COM FASTAPI
# ====================
app = FastAPI(
    title="Agentic RAG API with CrewAI",
    description="Este é um agente rag que analisa o currículo profissional e responde perguntas sobre ele.",
    version="1.0.0",
)
configure(app, chat_app)


if __name__ == "__main__":
    print("🚀 Iniciando servidor ReactPy...")
    print("📍 Acesse: http://localhost:8000")
    print("🛑 Pressione Ctrl+C para encerrar\n")

    # Usa string de importação "modulo:variavel" para permitir reload:
    uvicorn.run("ui:app", host="0.0.0.0", port=8000, reload=True)
