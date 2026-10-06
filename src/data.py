"""Load BEIR SciFact (5,183 abstracts, 300 test claims with relevance labels)."""
import csv, io, json, os, urllib.request, zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download():
    if os.path.exists(os.path.join(DIR, "scifact", "corpus.jsonl")):
        return
    os.makedirs(DIR, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall(DIR)


def load():
    download()
    p = os.path.join(DIR, "scifact")
    corpus = {}
    for l in open(os.path.join(p, "corpus.jsonl"), encoding="utf8"):
        d = json.loads(l); corpus[d["_id"]] = (d["title"] + ". " + d["text"]).strip()
    queries = {}
    for l in open(os.path.join(p, "queries.jsonl"), encoding="utf8"):
        d = json.loads(l); queries[d["_id"]] = d["text"]
    qrels = {}
    for r in csv.DictReader(open(os.path.join(p, "qrels", "test.tsv")), delimiter="\t"):
        if int(r["score"]) > 0:
            qrels.setdefault(r["query-id"], set()).add(r["corpus-id"])
    queries = {q: queries[q] for q in qrels}
    return corpus, queries, qrels
