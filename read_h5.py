import h5py
import numpy as np

def load_h5_embeddings(file_path, id_key='id', embedding_key='embedding'):
    """
    Load IDs and embeddings from an HDF5 file.

    Parameters:
    - id_key: Dataset name for the IDs inside the HDF5 file.
    - embedding_key: Dataset name for the embeddings inside the HDF5 file.

    Returns:
    - ids: Numpy array of IDs (as strings).
    - embeddings: Numpy array of embeddings (as float32).
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.ascontiguousarray(f[embedding_key], dtype=np.float32)

    print(f"Loaded {len(ids)} embeddings.")
    return ids, embeddings



if __name__ == "__main__":
    # usage:
    file_path = 'marco-data/msmarco_passages_embeddings_subset.h5'
    p_ids, p_embeddings = load_h5_embeddings('marco-data/msmarco_passages_embeddings_subset.h5')
    q_ids, q_embeddings = load_h5_embeddings('marco-data/msmarco_queries_dev_eval_embeddings.h5')

    # print(f"Passage id: {p_ids[0]}, Passage embedding:{p_embeddings[0]}")
    # print(f"Query id: {q_ids[0]}, Query embedding:{q_embeddings[0]}")
    print(f"Ids Shape {p_ids.shape}, Embedding shape: {p_embeddings.shape}, Embeddings Type: {p_embeddings.dtype}")
    print(f"Query Shape {q_ids.shape}, Query shape: {q_embeddings.shape}, query embedding Type: {p_embeddings.dtype}")
