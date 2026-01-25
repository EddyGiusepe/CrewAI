"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script ui.py
============
This script contains the chat interface styled like ChatGPT
with dark modern theme.
For the creation of this interface, we use the ReactPy framework.

Run
===
uv run ui.py

Access: http://localhost:8000
"""

import asyncio

import uvicorn
from application import ask_question  # Import the RAG agent from application.py
from fastapi import FastAPI
from reactpy import component, hooks, html
from reactpy.backend.fastapi import configure

# =======================
# CSS STYLES (Dark Theme)
# =======================
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


# ==========
# COMPONENTS
# ==========
@component
def chat_message(role: str, content: str):
    """Render a single message of the chat."""
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

    role_text = "You" if is_user else "🤖 Assistant"

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
    """Animated loading indicator."""
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
                "Assistant",
            ),
            html.div("Processing your question..."),
        ),
    )


@component
def chat_input(on_send, is_loading: bool):
    """Input field to send messages."""
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
                "placeholder": "Enter your question about the curriculum...",
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
    """Application header."""
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
            "Resume Analysis Agent",
        ),
        # html.p(
        #    {"style": subtitle_style},
        #    "Faça perguntas sobre o currículo profissional",
        # ),
    )


@component
def welcome_message():
    """Welcome message when there are no messages."""
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
        html.h2({"style": title_style}, "Welcome to the Resume Analysis Agent!"),
        html.p(
            {"style": text_style},
            "I am here to help you with information about the professional curriculum.",
        ),
        html.div(
            {"style": suggestions_style},
            html.div(
                {"style": suggestion_style},
                "💡 Example 1: About who is talking about the professional curriculum?",
            ),
            html.div(
                {"style": suggestion_style},
                "💡 Example 2: What are the technical skills of this professional?",
            ),
            html.div(
                {"style": suggestion_style},
                "💡 Example 3: What is the academic formation of this professional?",
            ),
        ),
    )


@component
def chat_app():
    """Main component of the chat."""
    messages, set_messages = hooks.use_state([])
    is_loading, set_is_loading = hooks.use_state(False)
    pending_question, set_pending_question = hooks.use_state(None)

    @hooks.use_effect(dependencies=[pending_question])
    async def process_pending_question():
        """Process the pending question when it changes."""
        if pending_question is None:
            return

        try:
            # Execute the ask_question function in a separate thread:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, ask_question, pending_question)
            response = str(result)
        except Exception as e:
            response = f"Error processing question: {e!s}"

        # Add the assistant's response:
        assistant_message = {"role": "assistant", "content": response}
        set_messages(lambda prev: [*prev, assistant_message])
        set_is_loading(False)
        set_pending_question(None)

    def handle_send(message: str):
        """Handle the sending of a new message."""
        # Add the user's message:
        new_user_message = {"role": "user", "content": message}
        set_messages(lambda prev: [*prev, new_user_message])
        set_is_loading(True)
        set_pending_question(message)

    # Main container styles:
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

    # Global CSS styles for scrollbar and font:
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

    # Render the messages:
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


# ======================
# EXECUTION WITH FASTAPI
# ======================
app = FastAPI(
    title="Agentic RAG API with CrewAI",
    description="This is a RAG agent that analyzes the professional curriculum and answers questions about it.",
    version="1.0.0",
)
configure(app, chat_app)


if __name__ == "__main__":
    print("🚀 Starting ReactPy server...")
    print("📍 Access: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop\n")

    # Use string of import "module:variable" to allow reload:
    uvicorn.run("ui:app", host="0.0.0.0", port=8000, reload=True)
