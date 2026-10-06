import math


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=10):
    return len(set(ranked[:k]) & rel) / len(rel)


def mrr_at_k(ranked, rel, k=10):
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1 / (i + 1)
    return 0.0
