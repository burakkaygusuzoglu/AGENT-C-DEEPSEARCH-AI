"""
RAG Pipeline
=============
- Ingests documents from the corpus/ folder (txt, pdf, md)
- Embeds them into a Chroma vector store
- Exposes a retrieve() function used by the agent
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────
CORPUS_DIR   = Path(__file__).parent.parent / "corpus"
CHROMA_DIR   = Path(__file__).parent.parent / ".chroma_db"
COLLECTION   = "research_corpus"
TOP_K        = 5


def _get_embeddings():
    """Use GitHub Models for embeddings too."""
    token = os.getenv("GITHUB_TOKEN", "MISSING")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=token,
        openai_api_base="https://models.github.ai/inference",
    )


def ingest(force: bool = False) -> Chroma:
    """
    Load documents from corpus/, chunk them, embed, store in Chroma.
    Re-uses existing DB unless force=True.
    """
    embeddings = _get_embeddings()

    # Return existing DB if already built
    if CHROMA_DIR.exists() and not force:
        print(f"📦 Loading existing vector store from {CHROMA_DIR}")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    print(f"🔄 Ingesting corpus from {CORPUS_DIR} ...")

    # Load all .txt and .md files
    loader = DirectoryLoader(
        str(CORPUS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(
            f"No documents found in {CORPUS_DIR}. "
            "Add .txt files to the corpus/ folder first."
        )

    print(f"   Loaded {len(docs)} documents")

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"   Split into {len(chunks)} chunks")

    # Embed and store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"✅ Vector store built with {len(chunks)} chunks")
    return vectorstore


def retrieve(query: str, k: int = TOP_K) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns a formatted string ready to be injected into a prompt.
    """
    try:
        vectorstore = ingest()
        docs = vectorstore.similarity_search(query, k=k)

        if not docs:
            return "No relevant documents found in the corpus."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Doc {i} | {Path(source).name}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"RAG retrieval failed: {e}"


# ── CLI: run ingestion standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    ingest(force=True)
