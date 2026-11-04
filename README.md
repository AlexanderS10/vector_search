# Vector Search & Re-ranking (Assignment #3)

This repo is for Assignment #3, where we compare our BM25 search engine from Assignment #2 against Dense Vector Retrieval (HNSW via `faiss`) and a Two-Stage BM25 + Vector Re-ranking system on the 1M MS-MARCO passage subset.

---

## 1. Data Setup (`marco-data/`)

Create a folder named `marco-data/` in the root of this repo and drop the files from the course Google Drive + `collection.tsv` from HW2 inside it:

```text
Vector_Search/
├── marco-data/
│   ├── collection.tsv                            # Full 8.8M passages from HW2
│   ├── msmarco_passages_subset.tsv               # 1,000,000 passage IDs in the subset
│   ├── msmarco_passages_embeddings_subset.h5     # 1M x 384 passage embeddings
│   ├── msmarco_queries_dev_eval_embeddings.h5    # Query embeddings (dev + eval)
│   ├── queries.dev.tsv                           # Query text for dev set (for BM25)
│   ├── queries.eval.tsv                          # Query text for eval sets (for BM25)
│   ├── qrels.dev.tsv                             # Dev relevance judgments (3 columns!)
│   ├── qrels.eval.one.tsv                        # TREC DL 2019 qrels (4 columns)
│   └── qrels.eval.two.tsv                        # TREC DL 2020 qrels (4 columns)
├── subset_processor.py
├── read_h5.py
└── hnsw_search.py
```

---

## 2. Python Environment Setup

> **Note:** Make sure to use **Python 3.12, 3.13, or 3.14** when creating the `.venv` (Python 3.15 does not have pre-built wheels for `h5py` or `faiss-cpu` yet and will fail trying to compile `libhdf5` from source yes I tried and it did not have it lol).

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy h5py faiss-cpu
```

---

## 3. Rebuilding the BM25 Index on the 1M Subset (System #1 Prerequisite)

The `msmarco_passages_subset.tsv` file only gives us the 1,000,000 external passage IDs, not the text. To filter the 8.8M `collection.tsv` down to just those 1,000,000 passages:

1. Run `subset_processor.py` from this repo:
   ```bash
   python subset_processor.py
   ```
   This outputs `collection_subset.tsv` (exact 1,000,000 lines matching the subset IDs).

2. Move `collection_subset.tsv` into our HW2 Rust repo (`Search_Index/data/raw/collection.tsv`), wipe the old 8.8M `data/runs/*` and `data/index/*` files so they don't mix, and re-run `parser` and `indexer`:
   ```bash
   # Inside Search_Index/
   rm -rf data/runs/* data/index/*
   mv ../Vector_Search/collection_subset.tsv data/raw/collection.tsv

   cargo run --release -p parser
   cargo run --release -p indexer
   ```
   No Rust code changes are needed for `parser` or `indexer`—the rebuilt index will have `total_docs = 1,000,000` and `avg_doc_len = 54.39`.

---

## 4. Running Dense Retrieval (System #2 — HNSW)

`hnsw_search.py` loads the `.h5` embeddings via `read_h5.py`, filters the query embeddings to the unique query IDs present in the `qrels` files, builds the `faiss.IndexHNSWFlat` index using inner product (`faiss.METRIC_INNER_PRODUCT`), and writes out standard 6-column TREC run files to `runs/`:

```bash
python hnsw_search.py          # baseline (4, 50, 50) on eval one only
python hnsw_search.py --all    # baseline (4, 50, 50) on eval one, eval two, and dev
python hnsw_search.py --sweep  # full 36-config HNSW sweep from (4, 50, 50) to (8, 200, 200)
```

`--sweep` builds 9 `(M, efConstruction)` graphs across `M = 4, 5, 6, 7, 8` and `efConstruction = 50, 100, 150, 200`, evaluates all 4 `efSearch` settings (`50, 100, 150, 200`) on `eval_one`, `eval_two`, and `dev` (writing `runs/hnsw_m{M}_efc_{efC}_efs{efS}_{label}.run`), and logs exact build time, total/graph RAM footprint, and per-query search latency to `eval/hnsw_stats.jsonl`.

---

## 5. Shared TREC Run Format & `trec_eval` Notes

So our benchmark script works across all 3 systems (BM25, HNSW, and Re-ranking), all systems should output rankings in standard 6-column TREC format inside `runs/`:

```text
<query_id>	Q0	<external_passage_id>	<rank>	<score>	<run_name>
```

### Heads up on `qrels.dev.tsv`:
- `qrels.eval.one.tsv` and `qrels.eval.two.tsv` have 4 columns (`qid \t 0 \t docid \t rel`).
- `qrels.dev.tsv` only has **3 columns** (`qid \t docid \t rel` — missing the `0` column in the 2nd position), so it needs that `0` column inserted before passing it to `trec_eval`.

---

## 6. Export BM25 candidates

From the Rust repo root, build `query` after creating the 1M subset index in
`data/index`. Set `VECTOR_SEARCH_DIR` to this repo's absolute path. The Rust
repo needs `crates/query/src/batch.rs` and its `main.rs`/`engine.rs` changes.

```bash
VECTOR_SEARCH_DIR=/absolute/path/to/vector_search
cargo build --release -p query

export_bm25() {
  target/release/query --batch \
    "$VECTOR_SEARCH_DIR/marco-data/queries.$1.tsv" \
    "$VECTOR_SEARCH_DIR/marco-data/qrels.$2.tsv" \
    "$VECTOR_SEARCH_DIR/ranks/$3.run" 1000 "$4"
}
export_bm25 dev dev bm25_dev or
export_bm25 eval eval.one bm25_eval_one or
export_bm25 eval eval.two bm25_eval_two or
export_bm25 dev dev bm25_and_dev and
export_bm25 eval eval.one bm25_and_eval_one and
export_bm25 eval eval.two bm25_and_eval_two and
```

The exporter writes directly to `ranks/`. The reranker can
read the `bm25_*.run` OR files or their `bm25_and_*.run` counterparts. AND may
return no candidates for a query. Missing IDs are null results. Run files
are ignored by Git.

---

## 7. Rerank candidate runs

`rerank.py` reranks passages in a six-column BM25 or HNSW `.run` file by
query-passage dot product. It takes a run file, not an index.

```text
python rerank.py CANDIDATE_RUN OUTPUT_RUN [--passage_vecs PASSAGE_H5] [--query_vecs QUERY_H5]
python rerank.py --all
```

`CANDIDATE_RUN` is the input `.run` file; `OUTPUT_RUN` is the new run path.
`--passage_vecs` defaults to `marco-data/msmarco_passages_embeddings_subset.h5`;
`--query_vecs` defaults to `marco-data/msmarco_queries_dev_eval_embeddings.h5`.
`--all` loads the `.h5` embeddings once into memory and reranks all 6 BM25 OR and AND runs (`eval_one`, `eval_two`, `dev`) into `runs/reranked_*.run` and `runs/reranked_and_*.run`.

Single-file examples:

```bash
python rerank.py ranks/bm25_eval_one.run runs/reranked_eval_one.run
python rerank.py runs/hnsw_m4_efc50_eval_one.run runs/reranked_hnsw_eval_one.run
```

---

## 8. Score retrieval runs

`score.py` needs the native `trec_eval` command on your PATH. On macOS, install
it with `brew install trec_eval`. Run it from this repo's directory:

```text
python score.py QRELS RUN [RUN ...]
python score.py --all
```

`QRELS` is a dev or eval qrels TSV; `RUN` is one or more six-column TREC run
files. For example:

```bash
python score.py marco-data/qrels.eval.one.tsv ranks/bm25_eval_one.run runs/hnsw_m4_efc50_eval_one.run runs/reranked_eval_one.run
```

`--all` scores BM25 OR, BM25 AND, all 36 HNSW sweep configurations, and both reranked runs for eval one,
eval two, and dev. Scores are pretty printed per run to terminal and results are appended to `eval/evals.jsonl`. Note: for now this is git ignored until we are ready
to submit the project so we don't append a bunch of dupes.

All sets get MRR@10 and Recall@100, eval gets NDCG@10/100. Dev gets MAP.
Dev's missing qrels column is added in a temporary file.

---

## 9. Experimental Benchmark Results (Data Record for Report)

### 9.1 HNSW Index Build Time & Memory Footprint (`1,000,000` vectors, `d = 384`)

Raw `float32` passage embeddings occupy fixed base storage of **`1,464.84 MB`** (`1,000,000 × 384 × 4` bytes). The HNSW multi-layer neighbor graph adds memory strictly as a linear function of **`M`** (~`+7.45 MB` per `+1` increase in `M`), while **`efConstruction`** affects build time and edge quality without increasing RAM.

| Graph Config `(M, efConstruction)` | Build Time (s) | Relative Build Time | Total Index RAM (MB) | HNSW Graph Overhead (MB) | Bytes / Vector (Graph Only) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`(M=4, efC=50)`** | **42.97 s** | 1.00× | 1,511.89 MB | **47.04 MB** | ~49.3 B |
| **`(M=4, efC=100)`** | **71.54 s** | 1.66× | 1,511.89 MB | **47.04 MB** | ~49.3 B |
| **`(M=4, efC=150)`** | **94.08 s** | 2.19× | 1,511.89 MB | **47.04 MB** | ~49.3 B |
| **`(M=5, efC=100)`** | **89.27 s** | 2.08× | 1,519.19 MB | **54.35 MB** | ~57.0 B |
| **`(M=5, efC=150)`** | **116.66 s** | 2.71× | 1,519.19 MB | **54.35 MB** | ~57.0 B |
| **`(M=6, efC=100)`** | **102.67 s** | 2.39× | 1,526.63 MB | **61.78 MB** | ~64.8 B |
| **`(M=6, efC=150)`** | **141.19 s** | 3.29× | 1,526.63 MB | **61.78 MB** | ~64.8 B |
| **`(M=7, efC=150)`** | **168.04 s** | 3.91× | 1,534.13 MB | **69.29 MB** | ~72.7 B |
| **`(M=8, efC=200)`** | **237.49 s** | 5.53× | 1,541.68 MB | **76.83 MB** | ~80.6 B |

---

### 9.2 TREC DL 2019 (`qrels.eval.one.tsv` — 43 Queries, Graded `0–3`)

| System / Config `(M, efC, efS)` | Latency (`ms/q`) | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System #1a: BM25 (`OR`, top 1000)** | — | 0.7942 | 0.5174 | 0.4517 | 0.5335 |
| **System #1b: BM25 (`AND`)** | — | 0.4070 | 0.0240 | 0.1293 | 0.0619 |
| **HNSW `(4, 50, 50)`** | 0.040 | 0.6705 | 0.3418 | 0.4535 | 0.4083 |
| **HNSW `(4, 50, 100)`** | 0.059 | 0.7915 | 0.4207 | 0.5446 | 0.5098 |
| **HNSW `(4, 50, 150)`** | 0.073 | 0.8380 | 0.4421 | 0.5753 | 0.5351 |
| **HNSW `(4, 50, 200)`** | 0.081 | 0.8845 | 0.4570 | 0.6111 | 0.5601 |
| **HNSW `(4, 100, 50)`** | 0.030 | 0.7217 | 0.4354 | 0.5294 | 0.5216 |
| **HNSW `(4, 100, 100)`** | 0.051 | 0.8845 | 0.5139 | 0.6247 | 0.6079 |
| **HNSW `(4, 100, 150)`** | 0.066 | 0.8845 | 0.5100 | 0.6247 | 0.6049 |
| **HNSW `(4, 100, 200)`** | 0.080 | 0.8845 | 0.5080 | 0.6247 | 0.6037 |
| **HNSW `(4, 150, 50)`** | 0.028 | 0.7217 | 0.4351 | 0.5231 | 0.5159 |
| **HNSW `(4, 150, 100)`** | 0.054 | 0.7915 | 0.4743 | 0.5665 | 0.5564 |
| **HNSW `(4, 150, 150)`** | 0.067 | 0.8380 | 0.4923 | 0.5862 | 0.5706 |
| **HNSW `(4, 150, 200)`** | 0.086 | 0.8612 | 0.4967 | 0.5961 | 0.5786 |
| **HNSW `(5, 100, 50)`** | 0.042 | 0.7868 | 0.4436 | 0.5740 | 0.5439 |
| **HNSW `(5, 100, 100)`** | 0.245 | 0.8798 | 0.4915 | 0.6258 | 0.5914 |
| **HNSW `(5, 100, 150)`** | 0.281 | 0.9264 | 0.5173 | 0.6564 | 0.6194 |
| **HNSW `(5, 100, 200)`** | 0.241 | 0.9496 | 0.5267 | 0.6657 | 0.6286 |
| **HNSW `(5, 150, 50)`** | 0.036 | 0.8147 | 0.4797 | 0.6037 | 0.5710 |
| **HNSW `(5, 150, 100)`** | 0.064 | 0.9078 | 0.5319 | 0.6605 | 0.6265 |
| **HNSW `(5, 150, 150)`** | 0.078 | 0.9310 | 0.5399 | 0.6698 | 0.6348 |
| **HNSW `(5, 150, 200)`** | 0.094 | 0.9310 | 0.5343 | 0.6698 | 0.6318 |
| **HNSW `(6, 100, 50)`** | 0.033 | 0.8372 | 0.4813 | 0.5980 | 0.5685 |
| **HNSW `(6, 100, 100)`** | 0.055 | 0.8837 | 0.5236 | 0.6367 | 0.6100 |
| **HNSW `(6, 100, 150)`** | 0.084 | 0.9302 | 0.5315 | 0.6647 | 0.6279 |
| **HNSW `(6, 100, 200)`** | 0.104 | 0.9302 | 0.5290 | 0.6647 | 0.6255 |
| **HNSW `(6, 150, 50)`** | 0.037 | 0.8837 | 0.5242 | 0.6274 | 0.6106 |
| **HNSW `(6, 150, 100)`** | 0.065 | 0.9070 | 0.5412 | 0.6491 | 0.6282 |
| **HNSW `(6, 150, 150)`** | 0.086 | 0.9302 | 0.5418 | 0.6713 | 0.6398 |
| **HNSW `(6, 150, 200)`** | 0.104 | 0.9302 | 0.5399 | 0.6713 | 0.6382 |
| **HNSW `(7, 150, 50)`** | 0.041 | 0.9070 | 0.5078 | 0.6588 | 0.6137 |
| **HNSW `(7, 150, 100)`** | 0.073 | 0.9302 | 0.5353 | 0.6793 | 0.6350 |
| **HNSW `(7, 150, 150)`** | 0.103 | 0.9535 | 0.5442 | 0.6886 | 0.6439 |
| **HNSW `(7, 150, 200)`** | 0.128 | 0.9535 | 0.5478 | 0.6926 | 0.6488 |
| **HNSW `(8, 200, 50)`** | 0.046 | 0.9535 | 0.5499 | 0.6994 | 0.6550 |
| **HNSW `(8, 200, 100)`** | 0.072 | 0.9535 | **0.5564** | 0.6987 | **0.6557** |
| **HNSW `(8, 200, 150)`** | 0.108 | 0.9535 | 0.5551 | 0.6987 | 0.6544 |
| **HNSW `(8, 200, 200)`** | 0.130 | 0.9535 | 0.5557 | 0.6987 | 0.6544 |
| **System #3a: BM25 (`OR`) + Vector Rerank** | — | **0.9651** | **0.5689** | **0.7053** | **0.6619** |
| **System #3b: BM25 (`AND`) + Vector Rerank** | — | 0.3845 | 0.0240 | 0.1323 | 0.0618 |

---

### 9.3 TREC DL 2020 (`qrels.eval.two.tsv` — 54 Queries, Graded `0–3`)

| System / Config `(M, efC, efS)` | Latency (`ms/q`) | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System #1a: BM25 (`OR`, top 1000)** | — | 0.8148 | 0.5251 | 0.4985 | 0.5144 |
| **System #1b: BM25 (`AND`)** | — | 0.3426 | 0.0250 | 0.1124 | 0.0619 |
| **HNSW `(4, 50, 50)`** | 0.039 | 0.6730 | 0.3621 | 0.4567 | 0.4056 |
| **HNSW `(4, 50, 100)`** | 0.050 | 0.8119 | 0.4432 | 0.5587 | 0.4955 |
| **HNSW `(4, 50, 150)`** | 0.072 | 0.8675 | 0.4472 | 0.5702 | 0.5031 |
| **HNSW `(4, 50, 200)`** | 0.079 | 0.8767 | 0.4602 | 0.5916 | 0.5227 |
| **HNSW `(4, 100, 50)`** | 0.034 | 0.7656 | 0.4649 | 0.5578 | 0.5106 |
| **HNSW `(4, 100, 100)`** | 0.051 | 0.8397 | 0.5099 | 0.6064 | 0.5562 |
| **HNSW `(4, 100, 150)`** | 0.070 | 0.8582 | 0.5141 | 0.6228 | 0.5654 |
| **HNSW `(4, 100, 200)`** | 0.085 | 0.8582 | 0.5100 | 0.6228 | 0.5634 |
| **HNSW `(4, 150, 50)`** | 0.040 | 0.7619 | 0.4444 | 0.5366 | 0.4968 |
| **HNSW `(4, 150, 100)`** | 0.053 | 0.8582 | 0.5244 | 0.6033 | 0.5657 |
| **HNSW `(4, 150, 150)`** | 0.066 | 0.8767 | 0.5286 | 0.6212 | 0.5733 |
| **HNSW `(4, 150, 200)`** | 0.082 | 0.8860 | 0.5448 | 0.6363 | 0.5870 |
| **HNSW `(5, 100, 50)`** | 0.063 | 0.8611 | 0.5484 | 0.6262 | 0.5846 |
| **HNSW `(5, 100, 100)`** | 0.068 | 0.9076 | 0.5855 | 0.6510 | 0.6154 |
| **HNSW `(5, 100, 150)`** | 0.092 | 0.9076 | 0.5820 | 0.6510 | 0.6138 |
| **HNSW `(5, 100, 200)`** | 0.093 | 0.9076 | 0.5797 | 0.6510 | 0.6124 |
| **HNSW `(5, 150, 50)`** | 0.039 | 0.9019 | 0.5841 | 0.6423 | 0.6098 |
| **HNSW `(5, 150, 100)`** | 0.062 | 0.9230 | 0.5995 | 0.6548 | 0.6228 |
| **HNSW `(5, 150, 150)`** | 0.087 | 0.9230 | 0.5958 | 0.6547 | 0.6215 |
| **HNSW `(5, 150, 200)`** | 0.115 | 0.9230 | 0.5934 | 0.6547 | 0.6201 |
| **HNSW `(6, 100, 50)`** | 0.044 | 0.9045 | 0.5932 | 0.6525 | 0.6194 |
| **HNSW `(6, 100, 100)`** | 0.060 | 0.9230 | 0.5970 | 0.6593 | 0.6251 |
| **HNSW `(6, 100, 150)`** | 0.085 | 0.9230 | 0.5914 | 0.6593 | 0.6227 |
| **HNSW `(6, 100, 200)`** | 0.114 | 0.9230 | 0.5905 | 0.6594 | 0.6219 |
| **HNSW `(6, 150, 50)`** | 0.052 | **0.9323** | 0.6128 | **0.6655** | 0.6356 |
| **HNSW `(6, 150, 100)`** | 0.082 | 0.9230 | 0.6093 | 0.6603 | 0.6337 |
| **HNSW `(6, 150, 150)`** | 0.087 | 0.9230 | 0.6061 | 0.6603 | 0.6331 |
| **HNSW `(6, 150, 200)`** | 0.112 | 0.9230 | 0.6043 | 0.6603 | 0.6318 |
| **HNSW `(7, 150, 50)`** | 0.049 | 0.9230 | **0.6141** | 0.6625 | **0.6391** |
| **HNSW `(7, 150, 100)`** | 0.067 | 0.9230 | 0.6123 | 0.6630 | 0.6386 |
| **HNSW `(7, 150, 150)`** | 0.112 | 0.9230 | 0.6082 | 0.6629 | 0.6362 |
| **HNSW `(7, 150, 200)`** | 0.157 | 0.9230 | 0.6087 | 0.6637 | 0.6369 |
| **HNSW `(8, 200, 50)`** | 0.044 | 0.9045 | 0.6019 | 0.6500 | 0.6259 |
| **HNSW `(8, 200, 100)`** | 0.092 | 0.9230 | 0.6112 | 0.6627 | 0.6377 |
| **HNSW `(8, 200, 150)`** | 0.102 | 0.9230 | 0.6068 | 0.6627 | 0.6353 |
| **HNSW `(8, 200, 200)`** | 0.131 | 0.9230 | 0.6056 | 0.6627 | 0.6343 |
| **System #3a: BM25 (`OR`) + Vector Rerank** | — | 0.9259 | 0.6111 | 0.6578 | 0.6339 |
| **System #3b: BM25 (`AND`) + Vector Rerank** | — | 0.3519 | 0.0250 | 0.1153 | 0.0638 |

---

### 9.4 MS-MARCO Dev (`qrels.dev.tsv` — 1,000 Queries, Binary `0/1`)

| System / Config `(M, efC, efS)` | Latency (`ms/q`) | MRR@10 | Recall@100 | MAP |
| :--- | :---: | :---: | :---: | :---: |
| **System #1a: BM25 (`OR`, top 1000)** | — | 0.3990 | 0.8050 | 0.4016 |
| **System #1b: BM25 (`AND`)** | — | 0.0951 | 0.1063 | 0.0924 |
| **HNSW `(4, 50, 50)`** | 0.029 | 0.2634 | 0.4327 | 0.2606 |
| **HNSW `(4, 50, 100)`** | 0.052 | 0.3098 | 0.5097 | 0.3074 |
| **HNSW `(4, 50, 150)`** | 0.080 | 0.3395 | 0.5602 | 0.3370 |
| **HNSW `(4, 50, 200)`** | 0.107 | 0.3616 | 0.5847 | 0.3586 |
| **HNSW `(4, 100, 50)`** | 0.033 | 0.3081 | 0.5172 | 0.3087 |
| **HNSW `(4, 100, 100)`** | 0.053 | 0.3763 | 0.6188 | 0.3739 |
| **HNSW `(4, 100, 150)`** | 0.074 | 0.4106 | 0.6703 | 0.4084 |
| **HNSW `(4, 100, 200)`** | 0.099 | 0.4279 | 0.6900 | 0.4249 |
| **HNSW `(4, 150, 50)`** | 0.031 | 0.3377 | 0.5550 | 0.3340 |
| **HNSW `(4, 150, 100)`** | 0.052 | 0.3994 | 0.6540 | 0.3955 |
| **HNSW `(4, 150, 150)`** | 0.072 | 0.4235 | 0.6887 | 0.4199 |
| **HNSW `(4, 150, 200)`** | 0.091 | 0.4426 | 0.7127 | 0.4383 |
| **HNSW `(5, 100, 50)`** | 0.045 | 0.3942 | 0.6478 | 0.3933 |
| **HNSW `(5, 100, 100)`** | 0.084 | 0.4417 | 0.7195 | 0.4397 |
| **HNSW `(5, 100, 150)`** | 0.110 | 0.4675 | 0.7562 | 0.4652 |
| **HNSW `(5, 100, 200)`** | 0.132 | 0.4860 | 0.7822 | 0.4837 |
| **HNSW `(5, 150, 50)`** | 0.037 | 0.4034 | 0.6667 | 0.4018 |
| **HNSW `(5, 150, 100)`** | 0.063 | 0.4536 | 0.7365 | 0.4523 |
| **HNSW `(5, 150, 150)`** | 0.090 | 0.4825 | 0.7772 | 0.4807 |
| **HNSW `(5, 150, 200)`** | 0.117 | 0.4931 | 0.7937 | 0.4908 |
| **HNSW `(6, 100, 50)`** | 0.061 | 0.4372 | 0.7133 | 0.4356 |
| **HNSW `(6, 100, 100)`** | 0.070 | 0.4822 | 0.7755 | 0.4808 |
| **HNSW `(6, 100, 150)`** | 0.102 | 0.5063 | 0.8080 | 0.5047 |
| **HNSW `(6, 100, 200)`** | 0.132 | 0.5204 | 0.8250 | 0.5190 |
| **HNSW `(6, 150, 50)`** | 0.042 | 0.4509 | 0.7330 | 0.4487 |
| **HNSW `(6, 150, 100)`** | 0.070 | 0.4871 | 0.7860 | 0.4852 |
| **HNSW `(6, 150, 150)`** | 0.102 | 0.5189 | 0.8270 | 0.5168 |
| **HNSW `(6, 150, 200)`** | 0.136 | 0.5330 | 0.8460 | 0.5311 |
| **HNSW `(7, 150, 50)`** | 0.047 | 0.4887 | 0.7799 | 0.4866 |
| **HNSW `(7, 150, 100)`** | 0.080 | 0.5279 | 0.8409 | 0.5256 |
| **HNSW `(7, 150, 150)`** | 0.108 | 0.5410 | 0.8564 | 0.5391 |
| **HNSW `(7, 150, 200)`** | 0.155 | 0.5516 | 0.8719 | 0.5495 |
| **HNSW `(8, 200, 50)`** | 0.055 | 0.5072 | 0.8111 | 0.5058 |
| **HNSW `(8, 200, 100)`** | 0.095 | 0.5427 | 0.8599 | 0.5414 |
| **HNSW `(8, 200, 150)`** | 0.142 | 0.5538 | 0.8809 | 0.5528 |
| **HNSW `(8, 200, 200)`** | 0.184 | **0.5603** | **0.8909** | **0.5592** |
| **System #3a: BM25 (`OR`) + Vector Rerank** | — | **0.5843** | **0.8933** | **0.5815** |
| **System #3b: BM25 (`AND`) + Vector Rerank** | — | 0.1022 | 0.1063 | 0.0988 |



---

## 10. BM25 batch export results

Hardware: MacBook Air (Apple M3, 8 CPU cores, 16 GB RAM).

Release build; one sequential run per set and mode, with snippets disabled and K = 1,000. Times include qrels loading, query filtering, search, and TREC file writing, but exclude index initialization. The index contains 1,000,000 passages (`avgdl = 54.344449`).

| Judged set | Queries | OR time | OR rows | AND time | AND rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev | 1,000 | 12.604 s | 998,299 | 0.033 s | 1,708 |
| Eval one | 43 | 0.543 s | 43,000 | 0.008 s | 128 |
| Eval two | 54 | 0.585 s | 54,000 | 0.009 s | 59 |

---

## 11. BM25 candidate count ablation

Results from `score.py` with `trec_eval -c` are recorded in `eval/evals.jsonl`. BM25 candidate counts are K=100, 200, 500, and 1,000; every recall value below is Recall@100. The fixed HNSW control retrieves 100 passages with `(M=8, efConstruction=200, efSearch=100)`.

### Raw BM25 scores

| Judged set | Mode | BM25 K | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 | MAP |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eval one | OR | 100 | 0.7942 | 0.5174 | 0.4517 | 0.5335 | — |
| Eval one | OR | 200 | 0.7942 | 0.5174 | 0.4517 | 0.5335 | — |
| Eval one | OR | 500 | 0.7942 | 0.5174 | 0.4517 | 0.5335 | — |
| Eval one | OR | 1,000 | 0.7942 | 0.5174 | 0.4517 | 0.5335 | — |
| Eval one | AND | 100 | 0.4070 | 0.0240 | 0.1293 | 0.0619 | — |
| Eval one | AND | 200 | 0.4070 | 0.0240 | 0.1293 | 0.0619 | — |
| Eval one | AND | 500 | 0.4070 | 0.0240 | 0.1293 | 0.0619 | — |
| Eval one | AND | 1,000 | 0.4070 | 0.0240 | 0.1293 | 0.0619 | — |
| Eval two | OR | 100 | 0.8148 | 0.5251 | 0.4985 | 0.5144 | — |
| Eval two | OR | 200 | 0.8148 | 0.5251 | 0.4985 | 0.5144 | — |
| Eval two | OR | 500 | 0.8148 | 0.5251 | 0.4985 | 0.5144 | — |
| Eval two | OR | 1,000 | 0.8148 | 0.5251 | 0.4985 | 0.5144 | — |
| Eval two | AND | 100 | 0.3426 | 0.0250 | 0.1124 | 0.0619 | — |
| Eval two | AND | 200 | 0.3426 | 0.0250 | 0.1124 | 0.0619 | — |
| Eval two | AND | 500 | 0.3426 | 0.0250 | 0.1124 | 0.0619 | — |
| Eval two | AND | 1,000 | 0.3426 | 0.0250 | 0.1124 | 0.0619 | — |
| Dev | OR | 100 | 0.3990 | 0.8050 | — | — | 0.4010 |
| Dev | OR | 200 | 0.3990 | 0.8050 | — | — | 0.4014 |
| Dev | OR | 500 | 0.3990 | 0.8050 | — | — | 0.4015 |
| Dev | OR | 1,000 | 0.3990 | 0.8050 | — | — | 0.4016 |
| Dev | AND | 100 | 0.0951 | 0.1063 | — | — | 0.0924 |
| Dev | AND | 200 | 0.0951 | 0.1063 | — | — | 0.0924 |
| Dev | AND | 500 | 0.0951 | 0.1063 | — | — | 0.0924 |
| Dev | AND | 1,000 | 0.0951 | 0.1063 | — | — | 0.0924 |

### Eval one (TREC DL 2019)

#### OR + rerank

| BM25 K | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.9612 | 0.5174 | 0.6899 | 0.5999 |
| 200 | 0.9806 | 0.5487 | 0.7135 | 0.6423 |
| 500 | 0.9690 | 0.5689 | 0.7101 | 0.6615 |
| 1,000 | 0.9651 | 0.5689 | 0.7053 | 0.6619 |

Fixed HNSW control (M=8, efConstruction=200, efSearch=100; top 100): MRR@10 0.9535, Recall@100 0.5567, NDCG@10 0.6987, NDCG@100 0.6559.

### Eval two (TREC DL 2020)

#### OR + rerank

| BM25 K | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.9204 | 0.5251 | 0.6418 | 0.5800 |
| 200 | 0.9204 | 0.5694 | 0.6556 | 0.6081 |
| 500 | 0.9241 | 0.6019 | 0.6591 | 0.6281 |
| 1,000 | 0.9259 | 0.6111 | 0.6578 | 0.6339 |

Fixed HNSW control (M=8, efConstruction=200, efSearch=100; top 100): MRR@10 0.9230, Recall@100 0.6113, NDCG@10 0.6627, NDCG@100 0.6377.

### Dev (MS MARCO)

#### OR + rerank

| BM25 K | MRR@10 | Recall@100 | MAP |
| ---: | ---: | ---: | ---: |
| 100 | 0.5664 | 0.8050 | 0.5615 |
| 200 | 0.5735 | 0.8420 | 0.5693 |
| 500 | 0.5792 | 0.8758 | 0.5762 |
| 1,000 | 0.5843 | 0.8933 | 0.5815 |

Fixed HNSW control (M=8, efConstruction=200, efSearch=100; top 100): MRR@10 0.5431, Recall@100 0.8599, MAP 0.5418.

### How to run

Export BM25 runs at K=100, 200, 500, and 1,000, rerank the OR candidates, then run `score.py` on the resulting runs. The README's export, rerank, and scoring sections give the commands.
