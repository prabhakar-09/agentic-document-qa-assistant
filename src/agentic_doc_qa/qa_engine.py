import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from agentic_doc_qa.chunker import split_documents
from agentic_doc_qa.document_loader import load_markdown_documents
from agentic_doc_qa.retriever import build_bm25_retriever

def load_hf_token() -> str:
    """
    Load Hugging Face token from the local .env file.
    """
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path)

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN was not found. Please add it to your .env file.")

    return hf_token

def retrieve_relevant_chunks(
    question: str,
    docs_dir: str = "data/raw_docs/fastapi/tutorial",
    top_k: int = 5,
):
    """
    Retrieve relevant FastAPI documentation chunks for a user question.
    """
    docs = load_markdown_documents(docs_dir)
    chunks = split_documents(docs)
    retriever = build_bm25_retriever(chunks, top_k=top_k)

    return retriever.invoke(question)

def format_context(chunks) -> str:
    """
    Format retrieved chunks into source-labeled context for answer generation.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "Unknown source")
        content = chunk.page_content

        context_parts.append(
            f"[Source {i}: {source}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)

def generate_answer(
    question: str,
    context: str,
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
) -> str:
    """
    Generate a source-grounded answer from the retrieved FastAPI documentation context.
    """
    hf_token = load_hf_token()

    client = InferenceClient(token=hf_token)

    prompt = f"""
You are a FastAPI documentation QA assistant.

Answer the user's question using only the provided documentation context.

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the context does not contain the answer, say:
  "I could not find this information in the FastAPI tutorial documentation."
- Write a concise answer in your own words.
- Include a Sources section with only the source file names used.

Question:
{question}

Documentation context:
{context}

Final answer format:

Answer:
<clear concise answer>

Sources:
- <source file>
"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return response.choices[0].message.content

def answer_question(question: str) -> str:
    """
    Answer a FastAPI documentation question using retrieval-augmented generation.
    """
    retrieved_chunks = retrieve_relevant_chunks(question)
    context = format_context(retrieved_chunks)

    return generate_answer(question, context)

if __name__ == "__main__":
    question = "How do I define a request body in FastAPI?"
    answer = answer_question(question)

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)