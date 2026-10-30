import time
import faiss
import numpy as np
from read_h5 import load_h5_embeddings


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

    print(f"Building HNSW index (M={m}, efConstruction={ef_construction}) on f{num_vectors} vectors...........")

    start_time = time.perf_counter()
    index.add(passage_embeddings)
    build_time_sec = time.perf_counter() - start_time

    print(f"Index built in {build_time_sec:.2f} seconds")
    return index, build_time_sec

def search_hnsw_index(index: faiss.IndexHNSWFlat, query_embeddings:np.ndarray, ef_search:int, k:int=100) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    #Sets the query time exploration depth
    index.hnsw.efSearch = ef_search
    num_queries = query_embeddings.shape[0]
    start_time = time.perf_counter()
    scores, internal_indices = index.search(query_embeddings, k)
    total_search_ms = (time.perf_counter() - start_time) * 1000.0
    avg_query_ms = total_search_ms/(num_queries if num_queries>0 else 0.0)
    return scores, internal_indices, total_search_ms, avg_query_ms