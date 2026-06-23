# app.py - CineBot com LangGraph + Groq + TMDB
# Versão: Final com roteador + fallback

import os
import urllib.parse
import json
from typing import TypedDict, List
import requests
import gradio as gr
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not GROQ_API_KEY or not TMDB_API_KEY:
    raise ValueError("Configure GROQ_API_KEY e TMDB_API_KEY no arquivo.env")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.7)

# 1. ESTADO DO AGENTE
class AgentState(TypedDict):
    user_input: str
    preferences: dict
    recommendations: List[dict]
    response: str
    fallback: bool

# 2. NÓS DO GRAFO

def entender_preferencias(state: AgentState):
    """Detecta se usuário falou de gênero ou citou um título específico."""
    prompt = f"""
    Analise a mensagem do usuário e retorne APENAS um JSON válido com 2 campos:
    1. "tipo": se ele citou um filme/série específico, retorne "titulo". Se falou de gostos/gêneros, retorne "genero".
    2. "valor": se for "titulo", retorne o nome do filme em inglês se souber. Se for "genero", retorne 1 frase resumindo o gosto.

    Exemplos:
    Usuario: gosto de comédia romântica leve
    Resposta: {{"tipo": "genero", "valor": "comédia romântica leve"}}

    Usuario: Um Lugar Chamado Notting Hill
    Resposta: {{"tipo": "titulo", "valor": "Notting Hill"}}

    Usuario: quero algo tipo Breaking Bad
    Resposta: {{"tipo": "titulo", "valor": "Breaking Bad"}}

    Usuario: {state['user_input']}
    Resposta:
    """
    resposta = llm.invoke(prompt).content.strip()
    try:
        # Limpa markdown caso o LLM invente
        resposta = resposta.replace("```json", "").replace("```", "").strip()
        dados = json.loads(resposta)
        return {"preferences": dados}
    except Exception as e:
        print(f"DEBUG - Erro no JSON: {e} | Resposta: {resposta}")
        # Fallback: assume que é gênero
        return {"preferences": {"tipo": "genero", "valor": state['user_input']}}

def buscar_tmdb(state: AgentState):
    """Busca no TMDB por título direto ou por keywords de gênero."""
    prefs = state["preferences"]

    if prefs["tipo"] == "titulo":
        # Busca direta por nome do filme/série
        query = prefs["valor"]
        print(f"DEBUG - Buscando título: {query}")
        url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={urllib.parse.quote(query)}&language=pt-BR"
    else:
        # Gera keywords pra busca por gênero
        query_prompt = f"""
        Gere 2 palavras-chave em INGLÊS para buscar no TMDB baseado em: {prefs["valor"]}
        Regras: APENAS 2 palavras, separadas por espaço, sem pontuação, sem aspas.
        Exemplos corretos: comedy romance | action thriller | sci-fi drama
        Resposta:
        """
        keywords = llm.invoke(query_prompt).content.strip()
        keywords = " ".join(keywords.split()[:2]).replace('"', '').replace("'", "").replace(",", "")
        print(f"DEBUG - Keywords pro TMDB: {keywords}")
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={urllib.parse.quote(keywords)}&language=pt-BR"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", [])[:5]:
            title = item.get("title") or item.get("name")
            overview = item.get("overview", "")
            poster = item.get("poster_path")
            if title and overview:
                results.append({
                    "title": title,
                    "overview": overview,
                    "poster": f"https://image.tmdb.org/t/p/w500{poster}" if poster else None
                })

        # Se não achou nada e era busca por gênero, ativa fallback
        fallback = not results and prefs["tipo"] == "genero"
        return {"recommendations": results, "fallback": fallback}

    except Exception as e:
        print(f"DEBUG - Erro TMDB: {e}")
        return {"recommendations": [], "fallback": True}

def formatar_resposta(state: AgentState):
    """Formata resposta final. Usa LLM se TMDB falhar."""
    recs = state["recommendations"]
    prefs = state["preferences"]

    # Fallback: TMDB não achou nada, deixa o LLM recomendar
    if state.get("fallback"):
        prompt = f"""
        O usuário quer: {prefs['valor']}
        O TMDB não retornou resultados úteis.
        Você é um cinéfilo especialista. Recomende 3 filmes ou séries em português do Brasil.
        Formato:
        **Nome do Filme/Série** - Por que combina, em 1 linha direta.
        Sem introdução, sem despedida. Só a lista.
        """
        resp = llm.invoke(prompt).content
        return {"response": resp}

    if not recs:
        return {"response": "Não achei nada com esse filtro. Me fala um gênero ou filme parecido que você curte?"}

    # Monta resposta com dados do TMDB
    lista = "\n".join([f"- **{r['title']}**: {r['overview'][:150]}..." for r in recs])
    prompt = f"""
    Usuário pediu: {prefs['valor']}
    Encontrei estas opções no TMDB:
    {lista}

    Escreva em português do Brasil, tom amigável de quem manja de cinema.
    Explique em 1 linha por que cada uma combina com o pedido.
    Use markdown **negrito** no nome dos filmes.
    """
    resp = llm.invoke(prompt).content
    return {"response": resp}

# 3. MONTAGEM DO GRAFO LANGGRAPH
workflow = StateGraph(AgentState)
workflow.add_node("entender", entender_preferencias)
workflow.add_node("buscar", buscar_tmdb)
workflow.add_node("responder", formatar_resposta)

workflow.set_entry_point("entender")
workflow.add_edge("entender", "buscar")
workflow.add_edge("buscar", "responder")
workflow.add_edge("responder", END)

app_graph = workflow.compile()

# 4. INTERFACE GRADIO
def chat_fn(message, history):
    try:
        state = {
            "user_input": message,
            "preferences": {},
            "recommendations": [],
            "response": "",
            "fallback": False
        }
        result = app_graph.invoke(state)
        return result["response"]
    except Exception as e:
        print(f"ERRO COMPLETO: {e}")
        return f"Deu ruim aqui 😅\nMotivo: {e}\n\nTenta de novo com outras palavras?"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="CineBot - Recomendador com LangGraph + Groq",
    description="Me fala o que você curte ou um filme parecido. Ex: 'comédia romântica tipo Notting Hill' ou 'ação com Tom Cruise'",
    examples=[
        "gosto de comédia romântica leve",
        "Um Lugar Chamado Notting Hill",
        "ficção científica tipo Blade Runner",
        "terror psicológico"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)