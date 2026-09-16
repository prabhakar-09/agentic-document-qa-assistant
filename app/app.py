import gradio as gr
from agentic_doc_qa.qa_engine import answer_question

def ask_fastapi_docs(question: str) -> str:
    """
    Send the user's question to the RAG backend and return the answer.
    """
    return answer_question(question)

demo = gr.Interface(
    fn=ask_fastapi_docs,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask a question about the FastAPI tutorial documentation...",
        label="Question",
    ),
    outputs=gr.Markdown(label="Answer"),
    title="Agentic Document QA Assistant",
    description=(
        "Ask questions about the official FastAPI tutorial documentation. "
        "Answers are generated only from retrieved documentation."
    ),
    examples=[
        ["How do I define a request body in FastAPI?"],
        ["How do I upload files in FastAPI?"],
        ["How do I run background tasks in FastAPI?"],
        ["How do I configure CORS in FastAPI?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )