"""Token-aware chunking strategies for PDF page text units.

Each strategy accepts a list of page-level text units. A text unit is expected
to contain at least doc_id, text, page_number, source_file, year, and doc_type.
"""

from __future__ import annotations

import re
from typing import Any

import tiktoken


ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text or ""))


def decode_tokens(tokens: list[int]) -> str:
    return ENCODING.decode(tokens).strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _slug_strategy(strategy: str) -> str:
    if strategy == "fixed_size":
        return "fixed"
    if strategy == "sentence_aware":
        return "sentence"
    return strategy


def _chunk_record(
    *,
    strategy: str,
    doc_id: str,
    index: int,
    text: str,
    pages: list[int],
    source_file: str,
    year: str,
    doc_type: str,
    source_span: str,
    section_heading: str = "",
) -> dict[str, Any]:
    page_start = min(pages)
    page_end = max(pages)
    strategy_id = _slug_strategy(strategy)
    return {
        "chunk_id": f"{strategy_id}::{doc_id}::c{index:04d}",
        "strategy": strategy_id,
        "doc_id": doc_id,
        "text": _normalize_text(text),
        "source_file": source_file,
        "year": year,
        "doc_type": doc_type,
        "page_number": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "source_span": source_span,
        "section_heading": section_heading,
        "token_count": count_tokens(text),
    }


def _group_units_by_doc(text_units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = {}
    for unit in text_units:
        text = _normalize_text(unit.get("text", ""))
        if not text:
            continue
        docs.setdefault(unit["doc_id"], []).append({**unit, "text": text})
    for units in docs.values():
        units.sort(key=lambda item: item.get("page_number", 0))
    return docs


def fixed_size(
    text_units: list[dict[str, Any]],
    max_tokens: int = 500,
    overlap_tokens: int = 75,
) -> list[dict[str, Any]]:
    """Split each document into fixed token windows, ignoring structure."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[dict[str, Any]] = []
    for doc_id, units in _group_units_by_doc(text_units).items():
        doc_tokens: list[int] = []
        token_pages: list[int] = []
        meta = units[0]
        for unit in units:
            tokens = ENCODING.encode(unit["text"])
            doc_tokens.extend(tokens)
            token_pages.extend([unit["page_number"]] * len(tokens))

        step = max_tokens - overlap_tokens
        chunk_index = 1
        for start in range(0, len(doc_tokens), step):
            end = min(start + max_tokens, len(doc_tokens))
            if start >= end:
                break
            window_tokens = doc_tokens[start:end]
            pages = sorted(set(token_pages[start:end]))
            text = decode_tokens(window_tokens)
            if text:
                chunks.append(
                    _chunk_record(
                        strategy="fixed",
                        doc_id=doc_id,
                        index=chunk_index,
                        text=text,
                        pages=pages,
                        source_file=meta["source_file"],
                        year=meta["year"],
                        doc_type=meta["doc_type"],
                        source_span=f"tokens:{start}-{end}",
                    )
                )
                chunk_index += 1
            if end == len(doc_tokens):
                break
    return chunks


def _sentences_for_unit(unit: dict[str, Any]) -> list[dict[str, Any]]:
    text = _normalize_text(unit["text"])
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences and text:
        sentences = [text]
    return [
        {
            "text": sentence,
            "page_number": unit["page_number"],
            "source_file": unit["source_file"],
            "year": unit["year"],
            "doc_type": unit["doc_type"],
        }
        for sentence in sentences
    ]


def _split_oversized_sentence(sentence: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    tokens = ENCODING.encode(sentence["text"])
    if len(tokens) <= max_tokens:
        return [sentence]
    pieces: list[dict[str, Any]] = []
    for start in range(0, len(tokens), max_tokens):
        text = decode_tokens(tokens[start : start + max_tokens])
        if text:
            pieces.append({**sentence, "text": text})
    return pieces


def sentence_aware(
    text_units: list[dict[str, Any]],
    max_tokens: int = 500,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """Split on sentence boundaries and group sentences near max_tokens."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    chunks: list[dict[str, Any]] = []
    for doc_id, units in _group_units_by_doc(text_units).items():
        sentences: list[dict[str, Any]] = []
        for unit in units:
            for sentence in _sentences_for_unit(unit):
                sentences.extend(_split_oversized_sentence(sentence, max_tokens))
        if not sentences:
            continue

        meta = units[0]
        chunk_index = 1
        current: list[dict[str, Any]] = []
        current_tokens = 0

        def emit(source_span: str) -> None:
            nonlocal chunk_index, current
            if not current:
                return
            text = " ".join(item["text"] for item in current)
            pages = sorted({item["page_number"] for item in current})
            chunks.append(
                _chunk_record(
                    strategy="sentence",
                    doc_id=doc_id,
                    index=chunk_index,
                    text=text,
                    pages=pages,
                    source_file=meta["source_file"],
                    year=meta["year"],
                    doc_type=meta["doc_type"],
                    source_span=source_span,
                )
            )
            chunk_index += 1

        start_sentence = 0
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = count_tokens(sentence["text"])
            if current and current_tokens + sentence_tokens > max_tokens:
                emit(f"sentences:{start_sentence}-{sentence_index}")
                current = current[-overlap_sentences:] if overlap_sentences else []
                start_sentence = max(0, sentence_index - len(current))
                current_tokens = sum(count_tokens(item["text"]) for item in current)
                if current and current_tokens + sentence_tokens > max_tokens:
                    current = []
                    start_sentence = sentence_index
                    current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens

        emit(f"sentences:{start_sentence}-{len(sentences)}")
    return chunks


HEADING_PATTERNS = [
    re.compile(r"^\s*(?:chapter|section|part)\s+[\w\dIVXLC]+[:.\-\s]", re.I),
    re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9, &'()/\-]{3,}$"),
    re.compile(r"^\s*[A-Z][A-Z0-9, &'()/\-]{6,}$"),
    re.compile(r"^\s*(?:NOTICE|DIRECTORS'? REPORT|CORPORATE GOVERNANCE|INDEPENDENT AUDITOR|BALANCE SHEET|STATEMENT OF PROFIT|CASH FLOW)", re.I),
]


def _looks_like_heading(line: str) -> bool:
    clean = line.strip()
    if not clean or len(clean) > 140:
        return False
    return any(pattern.search(clean) for pattern in HEADING_PATTERNS)


def _sections_for_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "document_start"
    current_lines: list[str] = []
    current_pages: set[int] = set()

    def flush() -> None:
        nonlocal current_lines, current_pages, current_heading
        text = _normalize_text(" ".join(current_lines))
        if text:
            sections.append(
                {
                    "heading": current_heading,
                    "text": text,
                    "pages": sorted(current_pages),
                }
            )
        current_lines = []
        current_pages = set()

    for unit in units:
        for raw_line in (unit.get("raw_text") or unit["text"]).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _looks_like_heading(line) and current_lines:
                flush()
                current_heading = line
            elif _looks_like_heading(line):
                current_heading = line
            current_lines.append(line)
            current_pages.add(unit["page_number"])
    flush()
    return sections


def semantic(
    text_units: list[dict[str, Any]],
    max_tokens: int = 700,
    overlap_tokens: int = 75,
) -> list[dict[str, Any]]:
    """Split by heading/section boundaries, with token windows for long sections."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[dict[str, Any]] = []
    for doc_id, units in _group_units_by_doc(text_units).items():
        meta = units[0]
        chunk_index = 1
        sections = _sections_for_units(units)
        for section_index, section in enumerate(sections):
            section_text = section["text"]
            section_tokens = ENCODING.encode(section_text)
            if not section_tokens:
                continue
            step = max_tokens - overlap_tokens
            for start in range(0, len(section_tokens), step):
                end = min(start + max_tokens, len(section_tokens))
                text = decode_tokens(section_tokens[start:end])
                if not text:
                    continue
                chunks.append(
                    _chunk_record(
                        strategy="semantic",
                        doc_id=doc_id,
                        index=chunk_index,
                        text=text,
                        pages=section["pages"] or [meta["page_number"]],
                        source_file=meta["source_file"],
                        year=meta["year"],
                        doc_type=meta["doc_type"],
                        source_span=f"section:{section_index};tokens:{start}-{end};heading:{section['heading']}",
                        section_heading=section["heading"],
                    )
                )
                chunk_index += 1
                if end == len(section_tokens):
                    break
    return chunks


def build_all_strategies(text_units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "fixed": fixed_size(text_units),
        "sentence": sentence_aware(text_units),
        "semantic": semantic(text_units),
    }
