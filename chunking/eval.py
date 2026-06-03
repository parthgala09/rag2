"""Evaluate chunk retrieval recall@5 and recall@10 for all strategies."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI
from pinecone import Pinecone

from chunking.env import load_env

load_env(ROOT / ".env")

CHUNKING_DIR = ROOT / "chunking"
TEST_SET_PATH = CHUNKING_DIR / "test_set.jsonl"
RESULTS_PATH = CHUNKING_DIR / "results.csv"
WINNER_PATH = CHUNKING_DIR / "winner.md"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_INDEX_NAME = "rag2-chunking-eval"
STRATEGIES = ("fixed", "sentence", "semantic")

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("test_set.jsonl is empty or missing. Run: python3 ingest.py --generate-test-set")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number} of {path}") from exc
    if not rows:
        raise RuntimeError("test_set.jsonl has no questions. Run: python3 ingest.py --generate-test-set")
    return rows


def openai_client() -> OpenAI:
    require_env("OPENAI_API_KEY")
    return OpenAI()


def pinecone_index():
    api_key = require_env("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        raise RuntimeError(f"Pinecone index does not exist: {index_name}. Run ingest.py first.")
    return pc.Index(index_name)


def embed_question(client: OpenAI, question: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    return response.data[0].embedding


def metadata_filter(item: dict[str, Any]) -> dict[str, Any]:
    query_filter: dict[str, Any] = {}
    if item.get("year"):
        query_filter["year"] = {"$eq": item["year"]}
    if item.get("doc_type"):
        query_filter["doc_type"] = {"$eq": item["doc_type"]}
    return query_filter


def query_ids(
    index,
    namespace: str,
    embedding: list[float],
    query_filter: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[str]:
    result = index.query(
        vector=embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=query_filter or None,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
    ids: list[str] = []
    for match in matches:
        ids.append(match["id"] if isinstance(match, dict) else match.id)
    return ids


def recall(retrieved_ids: list[str], truth_ids: list[str], k: int) -> int:
    if not truth_ids:
        return 0
    return int(bool(set(retrieved_ids[:k]) & set(truth_ids)))


def evaluate() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    test_set = load_test_set()
    client = openai_client()
    index = pinecone_index()
    rows: list[dict[str, Any]] = []

    for item in test_set:
        embedding = embed_question(client, item["question"])
        query_filter = metadata_filter(item)
        row: dict[str, Any] = {"question_id": item["question_id"]}
        for strategy in STRATEGIES:
            retrieved = query_ids(index, strategy, embedding, query_filter=query_filter, top_k=10)
            truth = item["ground_truth_chunk_ids"].get(strategy, [])
            row[f"{strategy}_recall_5"] = recall(retrieved, truth, 5)
            row[f"{strategy}_recall_10"] = recall(retrieved, truth, 10)
        rows.append(row)

    aggregates = {
        strategy: {
            "recall_5": mean(row[f"{strategy}_recall_5"] for row in rows),
            "recall_10": mean(row[f"{strategy}_recall_10"] for row in rows),
        }
        for strategy in STRATEGIES
    }
    return rows, aggregates


def write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    fieldnames = [
        "question_id",
        "fixed_recall_5",
        "fixed_recall_10",
        "sentence_recall_5",
        "sentence_recall_10",
        "semantic_recall_5",
        "semantic_recall_10",
    ]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_winner(aggregates: dict[str, dict[str, float]]) -> None:
    winner = max(STRATEGIES, key=lambda name: (aggregates[name]["recall_5"], aggregates[name]["recall_10"]))
    lines = [
        "# Chunking Evaluation Winner",
        "",
        "## Aggregate Scores",
        "",
        "| Strategy | Recall@5 | Recall@10 |",
        "| --- | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        lines.append(
            f"| {strategy} | {aggregates[strategy]['recall_5']:.3f} | {aggregates[strategy]['recall_10']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"## Recommended Chunker",
            "",
            f"Use `{winner}` for this corpus based on the highest recall@5, with recall@10 as the tie-breaker.",
            "",
            "## Rule of Thumb",
            "",
            "Use semantic chunking when reports have reliable headings and sections. Use sentence-aware chunking when prose is well formed but headings are inconsistent. Use fixed-size chunking as a baseline or when documents have little usable structure.",
        ]
    )
    WINNER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, aggregates = evaluate()
    write_results(rows)
    write_winner(aggregates)
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {WINNER_PATH}")


if __name__ == "__main__":
    main()
