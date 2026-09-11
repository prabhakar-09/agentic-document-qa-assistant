from smolagents import Tool
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class FastAPIDocsRetrieverTool(Tool):
    name = "fastapi_docs_retriever"
    description = (
        "Retrieves relevant information from the FastAPI tutorial documentation. "
        "Use this tool when the user asks about FastAPI concepts, request bodies, "
        "path parameters, query parameters, response models, dependencies, testing, "
        "background tasks, CORS, forms, files, security, or other FastAPI tutorial topics."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query related to FastAPI tutorial documentation.",
        }
    }

    output_type = "string"

    def __init__(self, retriever: BM25Retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def forward(self, query: str) -> str:
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant information found in the FastAPI tutorial documentation."

        result_text = "Retrieved FastAPI documentation:\n"

        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            start_index = doc.metadata.get("start_index", "Unknown")

            result_text += f"\n===== Result {i} =====\n"
            result_text += f"Source: {source}\n"
            result_text += f"Start index: {start_index}\n"
            result_text += f"Content:\n{doc.page_content}\n"

        return result_text

if __name__ == "__main__":
    from agentic_doc_qa.document_loader import load_markdown_documents
    from agentic_doc_qa.chunker import split_documents
    from agentic_doc_qa.retriever import build_bm25_retriever

    docs = load_markdown_documents("data/raw_docs/fastapi/tutorial")
    chunks = split_documents(docs)
    retriever = build_bm25_retriever(chunks, top_k=3)

    tool = FastAPIDocsRetrieverTool(retriever)

    query = "How do I define a request body in FastAPI?"
    result = tool.forward(query)

    print("Query:", query)
    print(result[:3000])