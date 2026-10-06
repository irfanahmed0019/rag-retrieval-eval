"""Compare BM25, dense (bge-small) and hybrid (reciprocal rank fusion) retrieval on SciFact."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load
from src.retrievers import Index, MODEL
from src.metrics import ndcg_at_k, recall_at_k, mrr_at_k


def main():
    corpus, queries, qrels = load()
    idx = Index(corpus, cache="data/vecs.npy")
    systems = {"BM25": idx.bm25_rank, "Dense (bge-small)": idx.dense_rank, "Hybrid (RRF)": idx.hybrid_rank}
    res = {}
    for name, fn in systems.items():
        n, r, m = [], [], []
        for q, text in queries.items():
            ranked = fn(text, 100)
            n.append(ndcg_at_k(ranked, qrels[q])); r.append(recall_at_k(ranked, qrels[q])); m.append(mrr_at_k(ranked, qrels[q]))
        # bootstrap 95% CI on nDCG@10
        rng = np.random.default_rng(0)
        boots = [np.mean(rng.choice(n, len(n))) for _ in range(1000)]
        res[name] = {"ndcg@10": round(float(np.mean(n)), 4), "ndcg@10_95ci": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                     "recall@10": round(float(np.mean(r)), 4), "mrr@10": round(float(np.mean(m)), 4)}
        print(name, res[name])
    out = {"dataset": "BEIR SciFact", "n_docs": len(corpus), "n_queries": len(queries), "dense_model": MODEL, "results": res}
    json.dump(out, open("reports/eval.json", "w"), indent=2)
    names = list(res)
    plt.bar(names, [res[n]["ndcg@10"] for n in names]); plt.ylabel("nDCG@10"); plt.title("SciFact retrieval quality")
    plt.savefig("reports/ndcg.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
