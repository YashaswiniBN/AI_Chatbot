# Agentic RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot using LangChain, LangGraph, OpenAI, Anthropic (Claude), Streamlit, ChromaDB, and PyPDF. It features an agent with multi-step reasoning, a RAG pipeline, and a Streamlit UI for document Q&A.

## Features
- PDF upload and ingestion
- RAG pipeline: PyPDFLoader, RecursiveCharacterTextSplitter, Chroma vector store, OpenAIEmbeddings
- Agent with nodes: decide_node, retrieve_node, answer_node, clarify_node
- Utility functions for LLM calls (OpenAI, Claude)
- Streamlit UI with chat history and context display

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage
- Upload PDF documents
- Ask questions in natural language
- The agent will decide to retrieve, clarify, or answer
- Responses are grounded in your documents

## Project Structure
- `app.py` — Streamlit UI and agent integration
- `rag_pipeline.py` — RAG pipeline utilities
- `agent.py` — Agent logic and nodes
- `llm_utils.py` — LLM utility functions

---
Replace API keys and secrets as needed for OpenAI/Anthropic.
