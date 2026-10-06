"""Small search API over the SciFact corpus (hybrid retrieval)."""
from fastapi import FastAPI
from src.data import load
from src.retrievers import Index

corpus, _, _ = load()
index = Index(corpus, cache="data/vecs.npy")
app = FastAPI(title="Hybrid retrieval demo")


@app.get("/search")
def search(q: str, k: int = 5, method: str = "hybrid"):
    fn = {"bm25": index.bm25_rank, "dense": index.dense_rank, "hybrid": index.hybrid_rank}[method]
    return [{"id": d, "text": corpus[d][:300]} for d in fn(q, k)]
