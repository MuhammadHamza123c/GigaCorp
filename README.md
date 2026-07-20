# GigaCorp RAG Agent — Customer Support Chatbot with Conversational Memory

A production-grade, enterprise-ready conversational RAG (Retrieval-Augmented Generation) agent built for GigaCorp customer support. Uses LangChain orchestration with Groq Cloud LLMs and a local FAISS vector store.

## Architecture

```
gigacorp-rag-agent/
├── data/                   # Raw FAQs knowledge base
├── src/
│   ├── config.py           # Environment & settings management
│   ├── agent/
│   │   ├── prompts.py      # System prompts & RAG templates
│   │   └── engine.py       # LangChain conversational retrieval chain
│   ├── database/
│   │   └── vector_store.py # FAISS index builder & loader
│   └── UI/
│       └── app.py          # Streamlit chat interface
└── tests/                  # Test stubs
```

## Quick Start

1. **Clone and enter the project:**

   ```bash
   cd gigacorp-rag-agent
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS / Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key:**

   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY=gsk_your_actual_key
   ```

5. **Run the application:**

   ```bash
   streamlit run src/UI/app.py
   ```

## Features

- **Conversational Memory** — Retains chat history across turns using LangChain's `ChatMessageHistory`.
- **Grounding on Knowledge Base** — Answers are strictly derived from the ingested FAQ documents.
- **Source Citations** — Every response includes document metadata (section title, line numbers) displayed in an expandable UI block.
- **Local Vector Store** — FAISS index built on first run with HuggingFace embeddings; no external embedding API needed.
- **Fully Free Tier** — Uses Groq Cloud (free credits) + open-source embeddings.

## Configuration

| Variable         | Required | Description                    |
|------------------|----------|--------------------------------|
| `GROQ_API_KEY`   | Yes      | Groq Cloud API key             |

## Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| Orchestration      | LangChain + LangChain Groq                     |
| LLM                | Groq Cloud (`llama-3.1-70b-versatile`)         |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2`        |
| Vector Store       | FAISS (local)                                   |
| UI                 | Streamlit                                       |
| Config             | `python-dotenv`                                 |
