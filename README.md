# CineBot - Recomendador de Filmes com LangGraph + GROQ

Assistente conversacional que recomenda filmes/séries usando LangGraph + Groq + TMDB.

**🚀 Demo Online:** https://upgraded-umbrella-v6ggw5q5p5rghxr77-7860.app.github.dev/

## Como Funciona

1. **Roteador LLM**: Detecta se você quer um gênero ou falou um título específico
2. **Busca TMDB**: Pesquisa por título direto ou gera 2 keywords em inglês
3. **Fallback Inteligente**: Se TMDB não achar, Llama-3.3 recomenda sozinho
4. **Resposta**: Explica em PT-BR por que cada filme combina com você

## Stack
- **Orquestração**: LangGraph
- **LLM**: Groq Llama-3.3-70b-versatile
- **API Externa**: TMDB
- **UI**: Gradio
- **Deploy**: GitHub Codespaces

## Rodando Local

1. Clone o repo
2. Crie `.env`: