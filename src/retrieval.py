"""Advanced retrieval: BM25 sparse search, dense+sparse fusion (RRF), and
cross-encoder reranking on top of the Chroma vector store."""

import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index built from whatever is currently in the Chroma collection."""

    def __init__(self, ids, documents, metadatas):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([tokenize(doc) for doc in documents]) if documents else None

    @classmethod
    def from_collection(cls, collection, where_filter=None):
        data = collection.get(where=where_filter, include=["documents", "metadatas"])
        return cls(data["ids"], data["documents"], data["metadatas"])

    def search(self, query, top_k):
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_indices[:top_k]:
            if scores[i] <= 0:
                continue
            results.append(
                {
                    "id": self.ids[i],
                    "document": self.documents[i],
                    "metadata": self.metadatas[i],
                    "score": float(scores[i]),
                }
            )
        return results


def dense_search(collection, query_embedding, top_k, where_filter=None):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )
    hits = []
    ids = results["ids"][0] if results["ids"] else []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results.get("distances") else [None] * len(ids)
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append({"id": doc_id, "document": doc, "metadata": meta, "distance": dist})
    return hits


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K):
    """Fuses several ranked lists of hits (each a list of dicts with an 'id' key)
    into one ranking using Reciprocal Rank Fusion."""
    scores = {}
    hit_by_id = {}
    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list):
            hit_by_id[hit["id"]] = hit
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank + 1)
    fused_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    return [dict(hit_by_id[doc_id], fused_score=scores[doc_id]) for doc_id in fused_ids]


def hybrid_search(
    collection,
    bm25_index,
    query,
    query_embedding,
    top_k,
    where_filter=None,
    use_bm25=True,
    candidate_multiplier=3,
):
    """Combines dense vector search with BM25 keyword search via RRF.
    Returns candidate_multiplier * top_k candidates for optional reranking."""
    n_candidates = top_k * candidate_multiplier
    dense_hits = dense_search(collection, query_embedding, n_candidates, where_filter)

    if not use_bm25:
        return dense_hits[:n_candidates]

    bm25_hits = bm25_index.search(query, n_candidates)
    fused = reciprocal_rank_fusion(dense_hits, bm25_hits)
    return fused[:n_candidates]


_reranker_cache = {}


def get_reranker():
    if "model" not in _reranker_cache:
        from sentence_transformers import CrossEncoder

        _reranker_cache["model"] = CrossEncoder(RERANKER_MODEL)
    return _reranker_cache["model"]


def rerank(query, hits, top_k):
    if not hits:
        return hits
    reranker = get_reranker()
    pairs = [(query, hit["document"]) for hit in hits]
    scores = reranker.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_k]
