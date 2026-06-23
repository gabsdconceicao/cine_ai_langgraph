# CineBot - Agente de Recomendação com LangGraph

Projeto iniciante para portfólio de CX Engineer.

## O que faz
Chat que entende seu gosto e recomenda filmes/séries usando TMDB.

## Stack gratuita
- LangGraph
- Google Gemini 1.5 Flash (API gratuita)
- TMDB API
- Gradio

## Como rodar no GitHub Codespaces
1. Code > Create codespace
2. No terminal: `pip install -r requirements.txt`
3. Copie `.env.example` para `.env` e cole suas chaves
4. `python app.py`
5. Abra a porta 7860 em Public

## Estrutura LangGraph
entender_preferencias -> buscar_tmdb -> formatar_resposta
