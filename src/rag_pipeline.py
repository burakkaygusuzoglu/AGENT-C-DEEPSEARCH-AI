"""
RAG Pipeline — Dynamic Self-Growing Corpus
==========================================
- Ingests documents from corpus/ folder on first run
- Web search results are automatically added to corpus (self-growing)
- Deduplication: content already in vector store is never re-added
- Smart ingest: tracks which files were already processed
"""

"""
RAG Pipeline — Dynamic Self-Growing Corpus
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CORPUS_DIR          = Path(__file__).parent.parent / "corpus"
CHROMA_DIR          = Path(__file__).parent.parent / ".chroma_db"
WEB_CORPUS_DIR      = Path(__file__).parent.parent / "corpus" / "web_learned"
COLLECTION          = "research_corpus"
TOP_K               = 5
DUPLICATE_THRESHOLD = 0.85

def _get_embeddings():
    token = os.getenv("GITHUB_TOKEN", "MISSING")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=token,
        openai_api_base="https://models.github.ai/inference",
    )

def _get_vectorstore(embeddings=None):
    if embeddings is None:
        embeddings = _get_embeddings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def ingest(force: bool = False):
    embeddings = _get_embeddings()
    WEB_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if force and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("🗑️  Cleared existing vector store")

    vectorstore = _get_vectorstore(embeddings)

    existing = vectorstore.get()
    indexed_sources = set()
    if existing and existing.get("metadatas"):
        for meta in existing["metadatas"]:
            if meta and meta.get("source"):
                indexed_sources.add(Path(meta["source"]).name)

    print(f"\n📂 Scanning corpus/ for new documents...")

    all_txts = list(CORPUS_DIR.rglob("*.txt"))
    new_docs = []
    skipped  = 0

    for txt_path in all_txts:
        if txt_path.name in indexed_sources and not force:
            skipped += 1
            continue
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            new_docs.extend(loader.load())
        except Exception as e:
            print(f"   ⚠️  Could not load {txt_path.name}: {e}")

    if skipped:
        print(f"   ⏭️  Skipped {skipped} already-indexed file(s)")

    if not new_docs:
        total = vectorstore._collection.count()
        print(f"   ✅ Nothing new. Vector store has {total} chunks.")
        return vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks   = splitter.split_documents(new_docs)
    print(f"   📄 {len(new_docs)} new documents → {len(chunks)} chunks")

    vectorstore.add_documents(chunks)
    total = vectorstore._collection.count()
    print(f"   ✅ Vector store now has {total} total chunks\n")
    return vectorstore

def retrieve(query: str, k: int = TOP_K) -> str:
    try:
        vectorstore = ingest()
        results     = vectorstore.similarity_search_with_score(query, k=k)

        if not results:
            return "No relevant documents found in the corpus."

        output = []
        for i, (doc, score) in enumerate(results, 1):
            source     = doc.metadata.get("source", "unknown")
            origin     = doc.metadata.get("origin", "corpus")
            similarity = round((1 - score) * 100, 1)
            label      = f"[Doc {i} | {Path(source).name} | {origin} | relevance: {similarity}%]"
            output.append(f"{label}\n{doc.page_content}")

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"RAG retrieval failed: {e}"

def learn_from_web(query: str, web_results: str) -> dict:
    if not web_results or "[Web search unavailable" in web_results:
        return {"added": 0, "skipped": 0}

    embeddings  = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)
    splitter    = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    blocks  = [b.strip() for b in web_results.split("\n\n---\n\n") if b.strip()]
    added   = 0
    skipped = 0

    for block in blocks:
        similar = vectorstore.similarity_search_with_score(block, k=1)
        if similar:
            _, top_score = similar[0]
            if (1 - top_score) >= DUPLICATE_THRESHOLD:
                skipped += 1
                continue

        content_id   = _content_hash(block)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename     = f"web_{timestamp}_{content_id}.txt"
        filepath     = WEB_CORPUS_DIR / filename
        safe_query   = query[:60].replace("/", "-").replace("\\", "-")
        file_content = f"Source: Web Search\nQuery: {safe_query}\nDate: {timestamp}\n\n{block}"
        filepath.write_text(file_content, encoding="utf-8")

        doc = Document(
            page_content=block,
            metadata={
                "source": str(filepath),
                "origin": "web_learned",
                "query":  query[:100],
                "date":   timestamp,
            }
        )
        chunks = splitter.split_documents([doc])
        vectorstore.add_documents(chunks)
        added += 1

    print(f"   🧠 Corpus update: +{added} new block(s), {skipped} duplicate(s) skipped")
    return {"added": added, "skipped": skipped}

def corpus_stats() -> dict:
    try:
        vectorstore  = _get_vectorstore()
        data         = vectorstore.get()
        total        = len(data["ids"]) if data and data.get("ids") else 0
        web_count    = 0
        corpus_count = 0
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                if meta and meta.get("origin") == "web_learned":
                    web_count += 1
                else:
                    corpus_count += 1
        return {
            "total_chunks":  total,
            "corpus_chunks": corpus_count,
            "web_chunks":    web_count,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    ingest(force=True)
    print(f"\n📊 Stats: {corpus_stats()}")