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

# TfidfVectorizer's built-in stop_words list only covers English. Jira issues
# in this workspace are frequently written in Spanish, so we add a small
# Spanish stopword list on top of sklearn's English one to keep topic labels
# from being dominated by filler words like "de", "el", "la", "se".
_SPANISH_STOPWORDS = frozenset(
    """
    de la que el en y a los del se las por un para con no una su al lo como
    mas pero sus le ya o este si porque esta entre cuando muy sin sobre
    tambien me hasta hay donde quien desde todo nos durante todos uno les
    ni contra otros ese eso ante ellos e esto mi antes algunos que unos yo
    otro otras otra el tanto esa estos mucho quienes nada muchos cual poco
    ella estar estas algunas algo nosotros mi mis tu te ti tu tus ellas
    nosotras vosotros vosotras os mio mia mios mias tuyo tuya tuyos tuyas
    suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra
    vuestros vuestras esos esas estoy esta estamos estais estan este estos
    estas fui fue fuimos fueron ser es soy eres somos sois son sea seas
    seamos seais sean siendo sido tener tengo tienes tiene tenemos teneis
    tienen para por hacia sobre bajo tras durante mediante segun
    """.split()
)


def _combined_stopwords() -> list[str]:
    """Spanish + English stopwords, since indexed Jira content in practice
    mixes Spanish prose with English technical terms."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(_SPANISH_STOPWORDS | ENGLISH_STOP_WORDS)


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
            stop_words=_combined_stopwords(),
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
