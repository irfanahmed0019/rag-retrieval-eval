import math
from src.metrics import ndcg_at_k, recall_at_k, mrr_at_k


def test_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == 1.0
    assert recall_at_k(["a", "b"], {"a", "b"}) == 1.0
    assert mrr_at_k(["a"], {"a"}) == 1.0


def test_miss():
    assert ndcg_at_k(["x"], {"a"}) == 0.0
    assert mrr_at_k(["x", "y"], {"a"}) == 0.0


def test_rank_discount():
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}), 1 / math.log2(3))
    assert mrr_at_k(["x", "a"], {"a"}) == 0.5
