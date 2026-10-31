import time
import faiss
import numpy as np
from read_h5 import load_h5_embeddings
import os


def get_queries_for_qrels(qrels_path: str, all_query_ids: np.ndarray, all_query_embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Filter the full query embeddings down to only the unque queries present in a file

    Params:
        qrels_path (str) = Path ot the TSV file
        all_query_ids (np.ndarray) = 1D array of all 202,185 query ids (strings)
        all_query_embeddings (np.ndarray) = 2D float32 array of shape (202185, 384)

    Returns:
        subset_qids (list[str]) = Ordered list of uniqye query ids form the qrels file
        subset_embeddings (np.ndarray) = Continguous 2D float32 array of shape (num_queries, 384)
    """

    unique_qids = []
    seen = set()
    with open(qrels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            qid = parts[0]
            if qid not in seen:
                seen.add(qid)
                unique_qids.append(qid)
    # Map every query id in the h5 file to its row index
    qid_to_row = {}
    for idx, qid in enumerate(all_query_ids):
        qid_to_row[qid] = idx

    # Collect the row indices for the queries we actually need
    matched_qids = []
    row_indices = []
    for qid in unique_qids:
        if qid in qid_to_row:
            matched_qids.append(qid)
            row_indices.append(qid_to_row[qid])

    subset_embeddings = np.ascontiguousarray(
        all_query_embeddings[row_indices], dtype=np.float32
    )
    return matched_qids, subset_embeddings


def build_hnsw_index(passage_embeddings: np.ndarray, m: int, ef_construction: int) -> tuple[faiss.IndexHNSWFlat, float]:
    """
    Build and in-memory FAISS HNSW index using the dot product similarity

    Params:
        passage_embeddings (np.ndarray) = 2D float32 array of shape (N, d)
        m (int) = Number of neighbor links each node is allowed to have
        ef_construction (int) = Dunamic candiate list size during graph construction

    Returns:
        index (faiss.IndexHNSWFlat) = Built HNSW index containing all N passage vectors
        build_time_sec (float) = wall clock time in seconds taht it takes to add all vectors
    """

    num_vectors, dim = passage_embeddings.shape
    # create the hnsw index configured for the dot product (inner product)
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    # set construction exploration depth BEFORE adding vectors
    index.hnsw.efConstruction = ef_construction #forgot to add this line lol not that anyone will read this
    
    print(
        f"Building HNSW index (M={m}, efConstruction={ef_construction}) on f{num_vectors} vectors...........")

    start_time = time.perf_counter()
    index.add(passage_embeddings)
    build_time_sec = time.perf_counter() - start_time

    print(f"Index built in {build_time_sec:.2f} seconds")
    return index, build_time_sec


def search_hnsw_index(index: faiss.IndexHNSWFlat, query_embeddings: np.ndarray, ef_search: int, k: int = 100) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run batch nearest neighbor seach over the index

    Params:
        index (faiss.IndexHNSWFlat)1 = Built faiss hnsw index
        query_embeddings (np.ndarray) = 2D float32 arrays fo shape (num_queries, d)
        ef_search (int) = Dynamic candidate list size duirng query time graph traversal
        k (int) = Number of top passages to retrieve per query (default = 100 for Recall@100/NDCG@100)

    Returns:
        scores (np.ndarray) = 2D float32 array of shape (num_queries, k) with dot product scores
        internal_indices (np.ndarray) = 2D int64 array of shape (num_queries, k) with row_indices
        total_search_ms (float) = Total search time in ms accross the query batch
        avg_query_ms (float) = Averrage search latencey in ms per query
    """
    # Sets the query time exploration depth
    index.hnsw.efSearch = ef_search
    num_queries = query_embeddings.shape[0]
    start_time = time.perf_counter()
    scores, internal_indices = index.search(query_embeddings, k)
    total_search_ms = (time.perf_counter() - start_time) * 1000.0
    avg_query_ms = total_search_ms/(num_queries if num_queries > 0 else 0.0)
    return scores, internal_indices, total_search_ms, avg_query_ms


def write_trec_run(output_path: str, query_ids: list[str], internal_indices: np.ndarray, scores: np.ndarray, passage_ids: np.ndarray, run_name: str) -> None:
    """
    Map FAISS internal row indices back to the external MSMARCO passage ids and write  results in a starndar 6 colum TREC run format for trec_eval

    format per line:
        query_id    0   external_passage_id    rank.   score    run_name

    Params:
        output_path (str) = File path to write the TREC run file
        query_ids (listp[str]) = Ordered list of query ids matching rows of scores/internal_indices
        internal_indices (np.ndarray) = 2D int64 array of shape (num_queries, k) from faiss
        scores (np.ndarray) = 2D float32 array of shape (num_queries, k) from faiss
        passage_ids (np.ndarray) = 1D array mapping internal row  i->external passage id string
        run_name (str) = Indentifier tag for this run 
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    num_queries, k = internal_indices.shape
    with open(output_path, "w") as f:
        for q_idx in range(num_queries):
            qid = query_ids[q_idx]
            for rank_idx in range(k):
                internal_row = internal_indices[q_idx, rank_idx]
                if internal_row < 0:  # if faiss couldn't reach k valid nodes
                    continue
                external_pid = passage_ids[internal_row]
                score = scores[q_idx, rank_idx]
                rank = rank_idx + 1
                f.write(
                    f"{qid}\tQ0\t{external_pid}\t{rank}\t{score:.6f}\t{run_name}\n")

if __name__ == "__main__":
    passage_ids, passage_embeddings = load_h5_embeddings(
        "marco-data/msmarco_passages_embeddings_subset.h5"
    )
    all_query_ids, all_query_embeddings = load_h5_embeddings(
        "marco-data/msmarco_queries_dev_eval_embeddings.h5"
    )
    
    #filter the queries for qrels.eval.one.tsv
    eval1_qids, eval1_1_emb = get_queries_for_qrels(
        "marco-data/qrels.eval.one.tsv", all_query_ids, all_query_embeddings
    )
    print(f"Filtered {len(eval1_qids)} unique queries for qrel.eval.one.tsv")
    #build baseiline hnsw index with m=4 and efconstruction 50
    index, build_sec = build_hnsw_index(passage_embeddings, m=4, ef_construction=50)
    #search top 100 passages per query wiht efSearch = 50
    scores, indices, total_ms, avg_ms = search_hnsw_index(index, eval1_1_emb, ef_search=50, k=100)
    print(
        f"Searched {len(eval1_qids)} queries in {total_ms:.2f} ms"
        f"({avg_ms:.3f} ms/query)"
    )
    
    #write the output 
    run_file = "runs/hnsw_m4_efc50_eval_one.run"
    write_trec_run(run_file, eval1_qids, indices, scores, passage_ids, run_name="hnsw_m4_efc50_efs50")
    print(f"Saved TREC run to file {run_file}")
    