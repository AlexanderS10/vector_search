"""scores eval 1, 2 and dev runs across retrieval types, print the run to a jsonl"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

def score(qrels_path: str, run_path: str, *flags: str) -> str:
    """run trec_eval on one run and return the raw scores"""
    # -c counts queries with no results instead of dropping them
    # fail on an eval error so we don't report partial scores
    return subprocess.run(
        ["trec_eval", "-c", *flags, qrels_path, run_path],
        check=True, capture_output=True, text=True,
    ).stdout


def prepare_qrels(qrels_path: str, directory: str) -> tuple[str, str, set[str]]:
    """add dev's missing 0 column and use map; eval qrels use ndcg"""
    # dev has 3 columns, the graded eval qrels already have 4
    with open(qrels_path) as source:
        dev = len(source.readline().split()) == 3
    # keep the original qrels file untouched
    if dev:
        converted = Path(directory) / "qrels.tsv"
        with open(qrels_path) as source, converted.open("w") as output:
            for line in source:
                qid, pid, rel = line.split()
                output.write(f"{qid}\t0\t{pid}\t{rel}\n")
        return str(converted), "map", {"recall_100", "mrr_10", "map"}
    return qrels_path, "ndcg_cut.10,100", {"recall_100", "mrr_10", "ndcg_10", "ndcg_100"}


def batch_eval(qrels_path: str, run_files: list[str]) -> None:
    """score each run, append it to evals.jsonl, and print the results"""
    with tempfile.TemporaryDirectory() as directory:
        prepared_qrels, metric, expected = prepare_qrels(qrels_path, directory)
        Path("eval").mkdir(exist_ok=True)
        # append so earlier evaluations stay in the jsonl file
        with open("eval/evals.jsonl", "a") as output:
            for file in run_files:
                results = score(prepared_qrels, file, "-m", "recall.100", "-m", metric)
                # cap reciprocal rank at 10 to get mrr@10
                results += score(prepared_qrels, file, "-M", "10", "-m", "recip_rank")
                metrics = {}
                # trec_eval's all rows are the averages across queries
                for line in results.splitlines():
                    name, query, value = line.split()
                    if query == "all":
                        name = {"recip_rank": "mrr_10", "ndcg_cut_10": "ndcg_10", "ndcg_cut_100": "ndcg_100"}.get(name, name)
                        metrics[name] = float(value)
                assert set(metrics) == expected, f"missing metrics for {file}: {metrics}"
                output.write(json.dumps({"qrels": qrels_path, "run": file, "metrics": metrics}) + "\n")
                print_report(file, metrics)

def print_report(file: str, metrics: dict[str, float]) -> None:
    """pretty print the scores in the terminal"""
    labels = {"mrr_10": "MRR@10", "recall_100": "Recall@100", "ndcg_10": "NDCG@10", "ndcg_100": "NDCG@100", "map": "MAP"}
    columns = [name for name in labels if name in metrics]
    print(f"\n{Path(file).name}")
    print("  ".join(f"{labels[name]:>12}" for name in columns))
    print("  ".join(f"{metrics[name]:>12.4f}" for name in columns))


def run_all() -> None:
    """score BM25, all HNSW sweep configs, and reranked runs for eval 1, eval 2 and dev"""
    hnsw_graphs = [
        (4, 50), (4, 100), (4, 150),
        (5, 100), (5, 150),
        (6, 100), (6, 150),
        (7, 150),
        (8, 200),
    ]
    efs_values = (50, 100, 150, 200)
    # each qrels set gets compared against its matching run files
    for label, qrel in (("eval_one", "eval.one"), ("eval_two", "eval.two"), ("dev", "dev")):
        hnsw_runs = [
            f"runs/hnsw_m{m}_efc_{efc}_efs{efs}_{label}.run"
            for m, efc in hnsw_graphs
            for efs in efs_values
        ]
        # use the existing runs without rebuilding the bm25 or hnsw indexes
        batch_eval(f"marco-data/qrels.{qrel}.tsv", [
            f"ranks/bm25_{label}.run",
            f"ranks/bm25_and_{label}.run",
            *hnsw_runs,
            f"runs/reranked_{label}.run",
            f"runs/reranked_and_{label}.run",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="score all eval1 / 2 / dev runs")
    parser.add_argument("qrels", nargs="?")
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    if args.all:
        run_all()
    elif args.qrels and args.runs:
        batch_eval(args.qrels, args.runs)
    else:
        parser.error("provide qrels and runs, or --all")
