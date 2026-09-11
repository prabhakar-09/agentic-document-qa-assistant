from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

def build_bm25_retriever(
    chunks: list[Document],
    top_k: int = 5,
) -> BM25Retriever:
    """
    Build a BM25 retriever from document chunks.
    """
    if not chunks:
        raise ValueError("Cannot build retriever because chunks list is empty.")

    return BM25Retriever.from_documents(chunks, k=top_k)

# Test
if __name__ == "__main__":
    from agentic_doc_qa.document_loader import load_markdown_documents
    from agentic_doc_qa.chunker import split_documents

    docs = load_markdown_documents("data/raw_docs/fastapi/tutorial")
    chunks = split_documents(docs)
    retriever = build_bm25_retriever(chunks, top_k=5)

    query = "How do I define a request body in FastAPI?"
    results = retriever.invoke(query)

    print("Query:", query)
    print("Results found:", len(results))

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Start index:", doc.metadata.get("start_index"))
        print("Preview:")
        print(doc.page_content[:500])