from agents.base_agent import AgentResult
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient

orch = Orchestrator(LLMClient(force_mock=True), session_id="fix3-check")

captured = {}
original_chat = orch.llm.chat
def spy_chat(**kwargs):
    captured["messages"] = kwargs["messages"]
    return original_chat(**kwargs)
orch.llm.chat = spy_chat

fake_results = [
    AgentResult(agent_name="RiskProfilingAgent", success=True,
                payload={"risk_class": "conservative", "confidence": 0.8}),
    AgentResult(agent_name="InvestmentAgent", success=False,
                error="No suitable products after filtering",
                payload={"status": "no_suitable_products",
                         "message": "No products in the catalogue are suitable for risk_class='conservative'."}),
]

orch._synthesise_response("what should I invest in?", fake_results, RoutingDecision.INVESTMENT)
prompt = captured["messages"][0]["content"]

assert "UNSUCCESSFUL" in prompt
assert "No products in the catalogue are suitable" in prompt
print("PASS — InvestmentAgent's denial message reached the synthesis prompt.\n")
print(prompt)
