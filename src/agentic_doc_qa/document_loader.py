from pathlib import Path
# from langchain_community.docstore.document import Document
from langchain_core.documents import Document

def load_markdown_documents(docs_dir: str | Path) -> list[Document]:
    """
    Load Markdown files from a directory and convert them into LangChain Document objects.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_path}")
    markdown_files = sorted(docs_path.rglob("*.md"))

    documents: list[Document] = []

    for file_path in markdown_files:
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        relative_path = file_path.relative_to(docs_path)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(relative_path),
                    "full_path": str(file_path),
                },
            )
        )

    return documents

if __name__ == "__main__":
    docs = load_markdown_documents("data/raw_docs/fastapi/tutorial")

    print("Documents loaded:", len(docs))
    print("Sample metadata:", docs[0].metadata)
    print("Sample content preview:")
    print(docs[0].page_content[:300])