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
        m (int) = Number of bi-directional links created per node in the HNSW graph
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
