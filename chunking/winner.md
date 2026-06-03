# Chunking Evaluation Winner

## Aggregate Scores

| Strategy | Recall@5 | Recall@10 |
| --- | ---: | ---: |
| fixed | 0.640 | 0.760 |
| sentence | 0.640 | 0.760 |
| semantic | 0.880 | 1.000 |

## Recommended Chunker

Use `semantic` for this corpus based on the highest recall@5, with recall@10 as the tie-breaker.

## Rule of Thumb

Use semantic chunking when reports have reliable headings and sections. Use sentence-aware chunking when prose is well formed but headings are inconsistent. Use fixed-size chunking as a baseline or when documents have little usable structure.
