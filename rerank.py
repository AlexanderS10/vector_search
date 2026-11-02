""" takes the results from a candidate run output and
reranks them using query and passage embeddings"""
from read_h5 import load_h5_embeddings
import argparse
from pathlib import Path

def make_bm_dict(path: str) -> dict[str, list[str]]:
    """key == qid, value == passage ids in run order. accepts path to a candidate run output"""
    candidates = {}
    with open(path) as source:
        for line in source:
            parts = line.split()
            if parts:
                candidates.setdefault(parts[0], []).append(parts[2])
    return candidates


def rerank_with_loaded(
    bm_dict: dict[str, list[str]],
    query_vectors,
    passage_vectors,
    query_row: dict[str, int],
    psg_row: dict[str, int],
) -> dict[str, list[tuple[str, float]]]:
    """rerank candidates using pre-loaded embeddings and id-to-row lookup tables"""
    results = {}
    for qid, pids in bm_dict.items():
        qvec = query_vectors[query_row[qid]]
        rows = [psg_row[pid] for pid in pids]
        # get dot product scores
        scores = passage_vectors[rows] @ qvec
        # sort by dot product, descending
        results[qid] = sorted(
            zip(pids, scores),
            key=lambda item: item[1],
            reverse=True,
        )
    return results


def rerank(bm_dict: dict[str, list[str]], passage_vec_file: str, query_vec_file: str) -> dict[str, list[tuple[str, float]]]:
    """find the vectors for the results in the candidate run file,
    calculate their dot product with the query vector,
    and sort by highest -> lowest dot product"""
    query_ids, query_vectors = load_h5_embeddings(query_vec_file)
    passage_ids, passage_vectors = load_h5_embeddings(passage_vec_file)
    # load 'er up with enumeration to map ids -> vector rows
    query_row = {qid: i for i, qid in enumerate(query_ids)}
    psg_row = {pid: i for i, pid in enumerate(passage_ids)}

    return rerank_with_loaded(bm_dict, query_vectors, passage_vectors, query_row, psg_row)

def write_results(results: dict[str, list[tuple[str, float]]], write_to: str) -> None:
    """persist reranking results as a TREC run tsv"""
    Path(write_to).parent.mkdir(parents=True, exist_ok=True)

    with open(write_to, 'w') as out:
        for qid, passages in results.items():
            for rank, (pid, score) in enumerate(passages, start = 1):
                out.write(
                    f"{qid}\tQ0\t{pid}\t{rank}\t{score:.9g}\tvector_reranked\n"
                )


def run_all(passage_vec_file: str, query_vec_file: str) -> None:
    """load embeddings once and rerank all BM25 OR and AND runs for eval_one, eval_two, and dev"""
    query_ids, query_vectors = load_h5_embeddings(query_vec_file)
    passage_ids, passage_vectors = load_h5_embeddings(passage_vec_file)
    query_row = {qid: i for i, qid in enumerate(query_ids)}
    psg_row = {pid: i for i, pid in enumerate(passage_ids)}

    for label in ("eval_one", "eval_two", "dev"):
        for src, dst in (
            (f"ranks/bm25_{label}.run", f"runs/reranked_{label}.run"),
            (f"ranks/bm25_and_{label}.run", f"runs/reranked_and_{label}.run"),
        ):
            bm_dict = make_bm_dict(src)
            reranked = rerank_with_loaded(bm_dict, query_vectors, passage_vectors, query_row, psg_row)
            write_results(reranked, dst)
            print(f"Reranked {src} -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="rerank all BM25 OR and AND runs for eval 1, 2, and dev")
    parser.add_argument("candidate_run", nargs="?", help = "six-column BM25 or HNSW run file")
    parser.add_argument("output_run", nargs="?", help = "output file path/name")
    parser.add_argument(
        "--passage_vecs",
        default = "marco-data/msmarco_passages_embeddings_subset.h5",
        help = "location of passage vectors"
    )
    parser.add_argument(
        "--query_vecs",
        default = "marco-data/msmarco_queries_dev_eval_embeddings.h5",
        help = "location of query embeddings"
    )
    args = parser.parse_args()

    if args.all:
        run_all(args.passage_vecs, args.query_vecs)
    elif args.candidate_run and args.output_run:
        bm_dict = make_bm_dict(args.candidate_run)
        reranked = rerank(bm_dict, args.passage_vecs, args.query_vecs)
        write_results(reranked, args.output_run)
    else:
        parser.error("provide candidate_run and output_run, or --all")
