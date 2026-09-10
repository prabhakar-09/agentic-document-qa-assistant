from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
documents:list[Document],
chunk_size: int=1000,
chunk_overlap: int=150,
) -> list[Document]:
    """
    Split loaded documents into smaller chunks for retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", ".", " ", ""],
    )

    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    from agentic_doc_qa.document_loader import load_markdown_documents

    docs = load_markdown_documents("data/raw_docs/fastapi/tutorial")
    chunks = split_documents(docs)

    print("Original documents:", len(docs))
    print("Processed chunks:", len(chunks))
    print("Sample chunk metadata:", chunks[0].metadata)
    print("Sample chunk preview:")
    print(chunks[0].page_content[:500])