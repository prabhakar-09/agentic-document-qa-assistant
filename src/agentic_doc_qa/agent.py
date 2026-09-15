import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel

from agentic_doc_qa.chunker import split_documents
from agentic_doc_qa.document_loader import load_markdown_documents
from agentic_doc_qa.retriever import build_bm25_retriever
from agentic_doc_qa.tools import FastAPIDocsRetrieverTool

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

def create_fastapi_docs_agent() -> CodeAgent:
    """
    Create a CodeAgent connected to the FastAPI documentation retriever tool.
    """
    hf_token = load_hf_token()

    docs = load_markdown_documents("data/raw_docs/fastapi/tutorial")
    chunks = split_documents(docs)
    retriever = build_bm25_retriever(chunks, top_k=3)

    fastapi_docs_tool = FastAPIDocsRetrieverTool(retriever)

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token,
    )

    agent = CodeAgent(
        tools=[fastapi_docs_tool],
        model=model,
        instructions="""
            You are an Agentic Document QA Assistant for FastAPI tutorial documentation.

            Use the fastapi_docs_retriever tool when the user asks about FastAPI documentation.

            Rules:
            1. Never pass the raw output of fastapi_docs_retriever directly to final_answer(...).
            2. You must read the retrieved documentation and write a concise answer in your own words.
            3. Use the fastapi_docs_retriever tool at most two times for one question.
            4. Do not use print(...) to show retrieved documents.
            5. Do not return raw retrieved documentation as the final answer.
            6. Summarize the retrieved documentation into a clear answer.
            7. Include the source file names used for the answer.
            8. Answer only using retrieved FastAPI documentation.
            9. If the retrieved documentation does not contain the answer, say:
            "I could not find this information in the FastAPI tutorial documentation."
            10. When the retrieved information is enough, call final_answer(...) immediately.
            """,
        max_steps=3,
    )

    return agent

if __name__ == "__main__":
    agent = create_fastapi_docs_agent()

    question = "How do I define a request body in FastAPI?"
    answer = agent.run(
        f"""
            Answer the following FastAPI documentation question:

            Question:
            {question}

            Requirements:
            - Use fastapi_docs_retriever to retrieve relevant documentation.
            - Do not return the raw retriever output.
            - Do not return snippets directly.
            - Write a concise answer in your own words based only on the retrieved documentation.
            - Include a final Sources section with only the source file names.
            - Final answer format:

            Answer:
            <clear concise answer>

            Sources:
            - <source file>
        """
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)