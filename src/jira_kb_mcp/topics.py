"""Topic detection over indexed issues via clustering + TF-IDF labeling.

We cluster on the embedding vectors already stored in LanceDB (no need to
recompute them), then label each cluster with its top TF-IDF terms so a
human gets a readable topic name without needing an LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_ISSUES_FOR_CLUSTERING = 5
TOP_TERMS_PER_TOPIC = 6


@dataclass
class Topic:
    topic_id: int
    label: str
    issue_keys: list[str]
    size: int


def _pick_k(n_samples: int) -> int:
    """Heuristic cluster count: roughly sqrt(n/2), bounded to [2, 25]."""
    k = max(2, int(np.sqrt(n_samples / 2)))
    return min(k, 25, n_samples)


def compute_topics(rows: list[dict]) -> tuple[dict[str, int], list[Topic]]:
    """Given rows with 'key', 'vector', 'summary', 'description', returns:
    - a mapping of issue key -> topic_id
    - a list of Topic summaries with a readable label

    Returns ({}, []) if there are too few issues to cluster meaningfully.
    """
    if len(rows) < MIN_ISSUES_FOR_CLUSTERING:
        return {}, []

    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    k = _pick_k(len(rows))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    key_to_topic = {row["key"]: int(label) for row, label in zip(rows, labels)}

    texts_by_cluster: dict[int, list[str]] = {}
    keys_by_cluster: dict[int, list[str]] = {}
    for row, label in zip(rows, labels):
        label = int(label)
        text = f"{row.get('summary', '')} {row.get('description', '')}"
        texts_by_cluster.setdefault(label, []).append(text)
        keys_by_cluster.setdefault(label, []).append(row["key"])

    topics = []
    for cluster_id, texts in texts_by_cluster.items():
        label_str = _label_cluster(texts)
        topics.append(
            Topic(
                topic_id=cluster_id,
                label=label_str,
                issue_keys=keys_by_cluster[cluster_id],
                size=len(keys_by_cluster[cluster_id]),
            )
        )

    topics.sort(key=lambda t: t.size, reverse=True)
    return key_to_topic, topics


def _label_cluster(texts: list[str]) -> str:
    """Labels a cluster with its top TF-IDF terms across the whole corpus
    of that cluster's issues."""
    if not texts or all(not t.strip() for t in texts):
        return "(sin contenido)"
    try:
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:TOP_TERMS_PER_TOPIC]
        top_terms = [terms[i] for i in top_indices if scores[i] > 0]
        return ", ".join(top_terms) if top_terms else "(sin términos distintivos)"
    except ValueError:
        return "(sin contenido)"
