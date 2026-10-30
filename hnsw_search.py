import time 
import faiss
import numpy as np
from read_h5 import load_h5_embeddings

def get_queries_for_qrels(qrels_path, all_query_ids, all_query_embeddings):
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
    
    #Collect the row indices for the queries we actually need 
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

