"""
RAG Pipeline
=============
- Ingests documents from the corpus/ folder (txt, pdf, md)
- Embeds them into a Chroma vector store
- Tracks ingested files via a manifest to support incremental updates
- Exposes a retrieve() function used by the agent
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_log = logging.getLogger("deep_research")

# ── Config ─────────────────────────────────────────────────────────────────────
CORPUS_DIR    = Path(__file__).parent.parent / "corpus"
CHROMA_DIR    = Path(__file__).parent.parent / ".chroma_db"
MANIFEST_FILE = CHROMA_DIR / "ingested_files.json"
COLLECTION    = "research_corpus"
TOP_K         = 5


def _get_embeddings() -> OpenAIEmbeddings:
    token = os.getenv("GITHUB_TOKEN", "MISSING")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=token,
        openai_api_base="https://models.github.ai/inference",
    )


# ── Manifest helpers (track which files are already embedded) ──────────────────
def _load_manifest() -> set[str]:
    if MANIFEST_FILE.exists():
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return set(data.get("files", []))
    return set()


def _save_manifest(files: set[str]) -> None:
    CHROMA_DIR.mkdir(exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps({"files": sorted(files)}, indent=2),
        encoding="utf-8",
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────
def ingest(force: bool = False, verbose: bool = False) -> Optional[Chroma]:
    """
    Load documents from corpus/, chunk, embed, and store in Chroma.

    - force=True  : wipe the DB and re-embed everything from scratch
    - verbose=True: print a detailed summary (used by the --ingest CLI flag)
    - Skips files already embedded (tracked via manifest)
    - Returns the Chroma vectorstore, or None if corpus is empty
    """
    if not CORPUS_DIR.exists():
        _log.warning("Corpus directory not found: %s", CORPUS_DIR)
        if verbose:
            print(f"⚠️  Corpus directory not found: {CORPUS_DIR}")
        return None

    all_files = sorted(CORPUS_DIR.glob("**/*.txt"))

    if not all_files:
        _log.warning("Corpus folder is empty — no documents to ingest.")
        if verbose:
            print("⚠️  Corpus folder is empty — add .txt files to corpus/")
        return None

    embeddings = _get_embeddings()

    # Force rebuild: wipe existing DB and manifest
    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        _log.info("Wiped existing vector store for full rebuild.")

    known     = _load_manifest() if not force else set()
    new_files = [f for f in all_files if str(f) not in known]
    skipped   = len(all_files) - len(new_files)

    # Nothing new to process — return existing store
    if not new_files:
        if CHROMA_DIR.exists():
            if verbose:
                print(f"\nIngest completed:\n - Documents: 0\n - Chunks:    0\n - Skipped:   {skipped}")
            return Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        return None

    # Load and chunk only new files
    splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []

    for file in new_files:
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            docs   = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            _log.info("Loaded %d chunks from %s", len(chunks), file.name)
        except Exception as e:
            _log.error("Skipping %s — load error: %s", file.name, e)
            if verbose:
                print(f"   ⚠️  Skipping {file.name}: {e}")

    if not all_chunks:
        _log.warning("No chunks produced from new documents.")
        if verbose:
            print("⚠️  No chunks created from new documents.")
        return None

    # Embed and persist
    if CHROMA_DIR.exists() and not force:
        # Incremental: add to existing collection
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        vectorstore.add_documents(all_chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
        )

    _save_manifest(known | {str(f) for f in new_files})
    _log.info("Ingest done: %d new docs, %d chunks, %d skipped", len(new_files), len(all_chunks), skipped)

    if verbose:
        print(f"\nIngest completed:")
        print(f" - Documents: {len(new_files)}")
        print(f" - Chunks:    {len(all_chunks)}")
        print(f" - Skipped:   {skipped}")

    return vectorstore


# ── Retrieve ───────────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns a formatted string ready for prompt injection.
    """
    try:
        vectorstore = ingest()

        if vectorstore is None:
            return "No documents available in the corpus."

        docs = vectorstore.similarity_search(query, k=k)

        if not docs:
            return "No relevant documents found in the corpus."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Doc {i} | {Path(source).name}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    except Exception as e:
        _log.error("RAG retrieval failed: %s", e, exc_info=True)
        return f"RAG retrieval failed: {type(e).__name__}"


# ── CLI: run ingestion standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    ingest(force=True, verbose=True)
