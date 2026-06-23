
# app.py - CineBot com LangGraph
import os
from typing import TypedDict, List
import requests
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not GEMINI_API_KEY or not TMDB_API_KEY:
    raise ValueError("Configure GEMINI_API_KEY e TMDB_API_KEY no arquivo .env")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)

class AgentState(TypedDict):
    user_input: str
    preferences: str
    recommendations: List[dict]
    response: str

def entender_preferencias(state: AgentState):
    prompt = f"""
    Você é um assistente de CX. Extraia os gostos de filmes ou séries do usuário em 1 frase.
    Entrada: {state['user_input']}
    """
    prefs = llm.invoke(prompt).content
    return {"preferences": prefs}

def buscar_tmdb(state: AgentState):
    prefs = state["preferences"]
    query_prompt = f"Transforme em 3 palavras-chave em inglês para buscar no TMDB: {prefs}"
    keywords = llm.invoke(query_prompt).content
    url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={keywords}&language=pt-BR"
    r = requests.get(url, timeout=10).json()
    results = []
    for item in r.get("results", [])[:5]:
        title = item.get("title") or item.get("name")
        overview = item.get("overview", "")
        if title and overview:
            results.append({"title": title, "overview": overview})
    return {"recommendations": results}

def formatar_resposta(state: AgentState):
    recs = state["recommendations"]
    if not recs:
        return {"response": "Não encontrei nada. Pode me dar mais detalhes do que gosta?"}
    lista = "\n".join([f"- {r['title']}: {r['overview'][:140]}..." for r in recs])
    prompt = f"""
    Usuário gosta de: {state['preferences']}
    Recomendações:
    {lista}
    Escreva em português do Brasil, tom amigável, explicando por que cada uma combina.
    """
    resp = llm.invoke(prompt).content
    return {"response": resp}

workflow = StateGraph(AgentState)
workflow.add_node("entender", entender_preferencias)
workflow.add_node("buscar", buscar_tmdb)
workflow.add_node("responder", formatar_resposta)
workflow.set_entry_point("entender")
workflow.add_edge("entender", "buscar")
workflow.add_edge("buscar", "responder")
workflow.add_edge("responder", END)
app_graph = workflow.compile()

def chat_fn(message, history):
    state = {"user_input": message, "preferences": "", "recommendations": [], "response": ""}
    result = app_graph.invoke(state)
    return result["response"]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="CineBot - Recomendador com LangGraph",
    description="Diga o que você curte. Exemplo: 'gosto de comédia leve tipo Ted Lasso'"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
