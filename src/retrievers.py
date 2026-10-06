import re
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

MODEL = "BAAI/bge-small-en-v1.5"
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


class Index:
    def __init__(self, corpus, cache=None):
        self.ids = list(corpus)
        self.bm25 = BM25Okapi([tok(corpus[i]) for i in self.ids])
        self.emb = TextEmbedding(MODEL, threads=2)
        if cache and __import__("os").path.exists(cache):
            self.vecs = np.load(cache)
        else:
            texts = [corpus[i] for i in self.ids]
            chunks = []
            for s in range(0, len(texts), 128):  # chunked to keep memory low
                print(f'embedding {s}/{len(texts)}', flush=True)
                chunks.append(np.array(list(self.emb.embed(texts[s:s + 128], batch_size=8)), dtype="float32"))
            self.vecs = np.concatenate(chunks)
            if cache:
                np.save(cache, self.vecs)

    def bm25_rank(self, q, k=100):
        s = self.bm25.get_scores(tok(q))
        return [self.ids[i] for i in np.argsort(-s)[:k]]

    def dense_rank(self, q, k=100):
        qv = next(iter(self.emb.query_embed(q)))
        s = self.vecs @ qv
        return [self.ids[i] for i in np.argsort(-s)[:k]]

    def hybrid_rank(self, q, k=100, rrf_k=60):
        score = {}
        for ranking in (self.bm25_rank(q, 100), self.dense_rank(q, 100)):
            for r, d in enumerate(ranking):
                score[d] = score.get(d, 0) + 1 / (rrf_k + r + 1)
        return sorted(score, key=score.get, reverse=True)[:k]
