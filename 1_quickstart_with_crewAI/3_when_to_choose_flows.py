#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Os fluxos são ideais quando
---------------------------

1. Você precisa de controle preciso sobre a execução - O fluxo de trabalho requer sequenciamento
                                                       exato e gerenciamento de estado
2. O aplicativo tem requisitos de estado complexos - Você precisa manter e transformar
                                                     o estado em várias etapas
3. Você precisa de resultados estruturados e previsíveis - O aplicativo requer resultados
                                                           consistentes e formatados
4. O fluxo de trabalho envolve lógica condicional - Diferentes caminhos precisam ser tomados
                                                    com base em resultados intermediários
5. Você precisa combinar IA com código procedural - A solução requer recursos de IA e programação tradicional

A seguir, um exemplo de um fluxo para Customer Support com processamento estruturado
-------------------------------------------------------------------------------------
"""
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel
from typing import List, Dict


# Definir o estado estruturado:
class SupportTicketState(BaseModel):
    ticket_id: str = ""
    customer_name: str = ""
    issue_description: str = ""
    category: str = ""
    priority: str = "medium"
    resolution: str = ""
    satisfaction_score: int = 0


class CustomerSupportFlow(Flow[SupportTicketState]):
    @start()
    def receive_ticket(self):
        # Em um aplicativo real, isso pode vir de uma API:
        self.state.ticket_id = "TKT-12345"
        self.state.customer_name = "Alex Johnson"
        self.state.issue_description = (
            "Não consigo acessar as funcionalidades premium após o pagamento"
        )
        return "Ticket recebido"

    @listen(receive_ticket)
    def categorize_ticket(self, _):
        # Use a chamada direta do LLM para categorização:
        from crewai import LLM

        llm = LLM(model="openai/gpt-4o-mini")

        prompt = f"""
        Categorizar o seguinte problema de suporte ao cliente em uma das seguintes categorias:
        - Billing
        - Account Access
        - Technical Issue
        - Feature Request
        - Other

        Problema: {self.state.issue_description}

        Retornar apenas o nome da categoria.
        """

        self.state.category = llm.call(prompt).strip()
        return self.state.category

    @router(categorize_ticket)
    def route_by_category(self, category):
        # Route to different handlers based on category
        return category.lower().replace(" ", "_")

    @listen("billing")
    def handle_billing_issue(self):
        # Handle billing-specific logic
        self.state.priority = "high"
        # More billing-specific processing...
        return "Billing issue handled"

    @listen("account_access")
    def handle_access_issue(self):
        # Handle access-specific logic
        self.state.priority = "high"
        # More access-specific processing...
        return "Access issue handled"

    # Additional category handlers...

    @listen("billing")
    @listen("account_access")
    @listen("technical_issue")
    @listen("feature_request")
    @listen("other")
    def resolve_ticket(self, resolution_info):
        # Final resolution step
        self.state.resolution = f"Problema (issue) resolvido: {resolution_info}"
        return self.state.resolution


# Executar o fluxo:
support_flow = CustomerSupportFlow()
result = support_flow.kickoff()

print(result)
