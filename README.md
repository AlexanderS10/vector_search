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
python hnsw_search.py
```

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

The exporter writes directly to `ranks/`; no copy is needed. The reranker can
read the `bm25_*.run` OR files or their `bm25_and_*.run` counterparts. AND may
return no candidates for a query; treat missing IDs as empty results. Run files
are ignored by Git, so regenerate or share them separately.

