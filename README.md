---
title: Agentic Document QA Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.12
app_file: app/app.py
pinned: false
---

# Agentic Document QA Assistant

A retrieval-augmented question-answering application built over the official FastAPI tutorial documentation.

The application retrieves relevant documentation using BM25 search and provides the retrieved context to a language model through the Hugging Face Inference API. The generated answer is grounded in the retrieved documentation and includes the relevant source files.

## Live Demo

The application is publicly deployed on Render.

**Live application:**  
[Open Agentic Document QA Assistant](https://agentic-document-qa-assistant.onrender.com/)

## What This Project Does

Users can ask natural-language questions about the FastAPI tutorial documentation, such as:

- How do I define a request body in FastAPI?
- How do I upload files?
- How do background tasks work?
- How do I configure CORS?

The system retrieves relevant documentation chunks and uses them as context for answer generation.

If the required information cannot be found in the documentation, the application returns a controlled fallback response instead of answering from outside knowledge.

## Architecture

```text
FastAPI Tutorial Markdown Files
            ↓
     Document Loader
            ↓
 Recursive Text Chunking
            ↓
       BM25 Retriever
            ↓
 Relevant Documentation Chunks
            ↓
 Hugging Face Inference API
            ↓
     Grounded Answer
            ↓
        Gradio UI

Key Features
- Retrieval-Augmented Generation (RAG)
- BM25 lexical retrieval
- Markdown-aware text preprocessing
- Recursive document chunking
- Source-aware answers
- Controlled fallback when documentation does not contain the answer
- Gradio web interface
- Hugging Face hosted model inference
- Public deployment using Render

Technology Stack
- Python 3.12
- LangChain
- BM25 / rank-bm25
- Hugging Face Hub / Inference API
- Gradio
- smolagents
- python-dotenv
- Render
- GitHub Codespaces

Project Structure

agentic-document-qa-assistant/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw_docs/
│   │   └── fastapi/tutorial/
│   └── processed_docs/
│
├── src/
│   └── agentic_doc_qa/
│       ├── document_loader.py
│       ├── chunker.py
│       ├── retriever.py
│       ├── qa_engine.py
│       ├── tools.py
│       └── agent.py
│
├── docs/
├── notebooks/
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md