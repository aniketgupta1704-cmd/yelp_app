"""Search functions — semantic (FAISS) and keyword (BM25-style)."""

import faiss
import numpy as np
import time


def semantic_search(query, embedding_model, index, df, top_k=5,
                    sentiment_filter=None, topic_filter=None):
    """Run FAISS-based semantic search and return top_k reviews."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search more than needed to allow for filtering
    scores, indices = index.search(query_embedding, top_k * 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        if sentiment_filter and row['sentiment'] != sentiment_filter:
            continue
        if topic_filter and topic_filter != 'All' and row['topic_label'] != topic_filter:
            continue
        results.append({
            'score': float(score),
            'sentiment': row['sentiment'],
            'topic': row['topic_label'],
            'stars': row['stars'],
            'text': row['text'],
            'date': row['date']
        })
        if len(results) == top_k:
            break
    return results


def keyword_search(query, df, top_k=5):
    """Simple keyword search — returns reviews containing all query terms."""
    query_terms = [t.lower() for t in query.split() if len(t) > 2]

    if not query_terms:
        return []

    # Score = number of query terms present
    text_lower = df['text'].str.lower()
    scores = np.zeros(len(df))
    for term in query_terms:
        scores += text_lower.str.contains(term, regex=False, na=False).astype(int)

    # Only include reviews matching at least one term
    matching_idx = np.where(scores > 0)[0]
    if len(matching_idx) == 0:
        return []

    # Sort by score, then by review length (shorter = more focused)
    matching_scores = scores[matching_idx]
    top_indices = matching_idx[np.argsort(-matching_scores)][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            'score': float(scores[idx] / len(query_terms)),  # normalize 0-1
            'sentiment': row['sentiment'],
            'topic': row['topic_label'],
            'stars': row['stars'],
            'text': row['text'],
            'date': row['date']
        })
    return results


def semantic_search_with_timing(query, embedding_model, index, df,
                                 top_k=5, sentiment_filter=None,
                                 topic_filter=None):
    """Semantic search that returns latency breakdown."""
    timings = {}

    # Time embedding
    t0 = time.time()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    timings['embedding_ms'] = (time.time() - t0) * 1000

    # Time FAISS search
    t0 = time.time()
    scores, indices = index.search(query_embedding, top_k * 3)
    timings['faiss_ms'] = (time.time() - t0) * 1000

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        if sentiment_filter and row['sentiment'] != sentiment_filter:
            continue
        if topic_filter and topic_filter != 'All' and row['topic_label'] != topic_filter:
            continue
        results.append({
            'score': float(score),
            'sentiment': row['sentiment'],
            'topic': row['topic_label'],
            'stars': row['stars'],
            'text': row['text'],
            'date': row['date']
        })
        if len(results) == top_k:
            break

    return results, timings