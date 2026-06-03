"""Extract PDFs, chunk them, embed chunks, and upsert into Pinecone.

Usage:
    python3 ingest.py
    python3 ingest.py --generate-test-set

Required environment:
    OPENAI_API_KEY
    PINECONE_API_KEY

Optional environment:
    PINECONE_INDEX_NAME=rag2-chunking-eval
    PINECONE_CLOUD=aws
    PINECONE_REGION=us-east-1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from chunking.env import load_env
from chunking.strategies import build_all_strategies


ROOT = Path(__file__).resolve().parent
CHUNKING_DIR = ROOT / "chunking"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_INDEX_NAME = "rag2-chunking-eval"

load_env(ROOT / ".env")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def infer_year(source_file: str, first_page_text: str = "") -> str:
    haystack = f"{source_file} {first_page_text}"
    ranges = re.findall(r"(20\d{2})\s*[-_/]\s*(\d{2})", haystack)
    if ranges:
        start, end = ranges[-1]
        return f"{start}-{end}"
    years = re.findall(r"20\d{2}", haystack)
    return years[-1] if years else "unknown"


def doc_id_for_pdf(path: Path, first_page_text: str = "") -> str:
    year = infer_year(path.name, first_page_text)
    base = slugify(path.stem)
    return slugify(f"{base}-{year}")


def pdf_paths(limit: int | None = None) -> list[Path]:
    paths = sorted(ROOT.glob("*.pdf"))
    return paths[:limit] if limit else paths


def extract_text_units(limit_docs: int | None = None) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for pdf_path in pdf_paths(limit_docs):
        reader = PdfReader(str(pdf_path))
        first_text = reader.pages[0].extract_text() or "" if reader.pages else ""
        year = infer_year(pdf_path.name, first_text)
        doc_id = doc_id_for_pdf(pdf_path, first_text)
        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", raw_text).strip()
            if not text:
                continue
            units.append(
                {
                    "doc_id": doc_id,
                    "source_file": pdf_path.name,
                    "year": year,
                    "doc_type": "annual_report",
                    "page_number": page_index,
                    "page_start": page_index,
                    "page_end": page_index,
                    "text": text,
                    "raw_text": raw_text,
                }
            )
    return units


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def openai_client() -> OpenAI:
    require_env("OPENAI_API_KEY")
    return OpenAI()


def pinecone_index():
    api_key = require_env("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while True:
            status = pc.describe_index(index_name).status
            ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
            if ready:
                break
            time.sleep(2)
    return pc.Index(index_name)


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 96) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def pinecone_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    text = chunk["text"]
    return {
        "chunk_id": chunk["chunk_id"],
        "strategy": chunk["strategy"],
        "doc_id": chunk["doc_id"],
        "source_file": chunk["source_file"],
        "year": chunk["year"],
        "doc_type": chunk["doc_type"],
        "page_number": int(chunk["page_number"]),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
        "source_span": chunk.get("source_span", ""),
        "section_heading": chunk.get("section_heading", ""),
        "token_count": int(chunk.get("token_count", 0)),
        "text": text[:40000],
    }


def upsert_strategy(index, client: OpenAI, strategy: str, chunks: list[dict[str, Any]]) -> None:
    for start in range(0, len(chunks), 96):
        batch = chunks[start : start + 96]
        embeddings = embed_texts(client, [chunk["text"] for chunk in batch])
        vectors = [
            {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": pinecone_metadata(chunk),
            }
            for chunk, embedding in zip(batch, embeddings, strict=True)
        ]
        index.upsert(vectors=vectors, namespace=strategy)


def choose_test_chunks(strategy_chunks: dict[str, list[dict[str, Any]]], count: int = 25) -> list[dict[str, Any]]:
    semantic_chunks = [
        chunk
        for chunk in strategy_chunks["semantic"]
        if len(chunk["text"]) >= 350 and not re.search(r"\b(table of contents|contents)\b", chunk["text"], re.I)
    ]
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in semantic_chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    selected: list[dict[str, Any]] = []
    doc_ids = sorted(by_doc)
    while len(selected) < count and any(by_doc.values()):
        for doc_id in doc_ids:
            if by_doc.get(doc_id):
                selected.append(by_doc[doc_id].pop(0))
                if len(selected) == count:
                    break
    return selected


def overlapping_chunk_ids(chunks: list[dict[str, Any]], doc_id: str, page_start: int, page_end: int) -> list[str]:
    ids = []
    for chunk in chunks:
        if chunk["doc_id"] != doc_id:
            continue
        overlaps = chunk["page_start"] <= page_end and chunk["page_end"] >= page_start
        if overlaps:
            ids.append(chunk["chunk_id"])
    return ids


def generate_question(client: OpenAI, chunk: dict[str, Any], question_id: str) -> dict[str, Any]:
    prompt = (
        "Create one concise retrieval evaluation question answerable only from the excerpt. "
        "Return strict JSON with keys question and answer_hint. "
        "The question should ask about a concrete fact, number, policy, role, or statement.\n\n"
        f"Document year: {chunk['year']}\n"
        f"Pages: {chunk['page_start']}-{chunk['page_end']}\n"
        f"Excerpt:\n{chunk['text'][:2500]}"
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_TESTSET_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return {
        "question_id": question_id,
        "question": data["question"],
        "answer_hint": data.get("answer_hint", ""),
    }


def write_test_set(strategy_chunks: dict[str, list[dict[str, Any]]], client: OpenAI) -> Path:
    CHUNKING_DIR.mkdir(exist_ok=True)
    selected = choose_test_chunks(strategy_chunks, 25)
    if len(selected) < 25:
        raise RuntimeError(f"Only found {len(selected)} suitable chunks for test-set generation")

    out_path = CHUNKING_DIR / "test_set.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for index, chunk in enumerate(selected, start=1):
            question_id = f"q{index:03d}"
            item = generate_question(client, chunk, question_id)
            item.update(
                {
                    "doc_id": chunk["doc_id"],
                    "year": chunk["year"],
                    "doc_type": chunk["doc_type"],
                    "page_number": chunk["page_number"],
                    "source_pages": [chunk["page_start"], chunk["page_end"]],
                    "ground_truth_chunk_ids": {
                        strategy: overlapping_chunk_ids(
                            chunks,
                            chunk["doc_id"],
                            chunk["page_start"],
                            chunk["page_end"],
                        )
                        for strategy, chunks in strategy_chunks.items()
                    },
                }
            )
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path


def write_chunk_manifest(strategy_chunks: dict[str, list[dict[str, Any]]]) -> Path:
    CHUNKING_DIR.mkdir(exist_ok=True)
    out_path = CHUNKING_DIR / "chunks_manifest.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for strategy in ("fixed", "sentence", "semantic"):
            for chunk in strategy_chunks[strategy]:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index annual-report PDF chunks into Pinecone.")
    parser.add_argument("--generate-test-set", action="store_true", help="Generate chunk-grounded 25-question test set.")
    parser.add_argument("--limit-docs", type=int, default=None, help="Only process the first N PDFs.")
    parser.add_argument("--skip-upsert", action="store_true", help="Build chunks/test set without writing to Pinecone.")
    args = parser.parse_args()

    text_units = extract_text_units(limit_docs=args.limit_docs)
    if not text_units:
        raise RuntimeError("No extractable PDF text found.")

    strategy_chunks = build_all_strategies(text_units)
    manifest = write_chunk_manifest(strategy_chunks)
    print(f"Wrote chunk manifest: {manifest}")
    for strategy, chunks in strategy_chunks.items():
        print(f"{strategy}: {len(chunks)} chunks")

    client = openai_client() if (not args.skip_upsert or args.generate_test_set) else None
    if not args.skip_upsert:
        assert client is not None
        index = pinecone_index()
        for strategy, chunks in strategy_chunks.items():
            print(f"Upserting {len(chunks)} {strategy} chunks...")
            upsert_strategy(index, client, strategy, chunks)

    if args.generate_test_set:
        assert client is not None
        path = write_test_set(strategy_chunks, client)
        print(f"Wrote test set: {path}")


if __name__ == "__main__":
    main()
