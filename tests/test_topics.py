import numpy as np

from jira_kb_mcp.topics import compute_topics


def _fake_rows(n_per_cluster: int = 6):
    """Builds two well-separated synthetic clusters so KMeans has an easy,
    deterministic job — this is a unit test for the plumbing, not for
    clustering quality."""
    rng = np.random.default_rng(42)
    rows = []
    cluster_centers = {
        0: (np.array([10.0, 10.0, 10.0], dtype=np.float32), "login timeout error"),
        1: (np.array([-10.0, -10.0, -10.0], dtype=np.float32), "payment gateway declined"),
    }
    counter = 0
    for cluster_id, (center, text) in cluster_centers.items():
        for _ in range(n_per_cluster):
            vector = center + rng.normal(scale=0.01, size=3).astype(np.float32)
            rows.append(
                {
                    "key": f"SEC-{counter}",
                    "vector": vector.tolist(),
                    "summary": text,
                    "description": text,
                }
            )
            counter += 1
    return rows


def test_compute_topics_too_few_rows_returns_empty():
    key_to_topic, topics = compute_topics([{"key": "SEC-1", "vector": [0, 0, 0], "summary": "x", "description": "y"}])
    assert key_to_topic == {}
    assert topics == []


def test_compute_topics_produces_clusters_and_labels():
    rows = _fake_rows()
    key_to_topic, topics = compute_topics(rows)

    assert len(key_to_topic) == len(rows)
    assert len(topics) >= 2

    for topic in topics:
        assert topic.size > 0
        assert isinstance(topic.label, str)
        assert topic.label != ""
        assert all(isinstance(k, str) for k in topic.issue_keys)


def test_compute_topics_every_key_covered_exactly_once():
    rows = _fake_rows()
    key_to_topic, topics = compute_topics(rows)

    keys_from_topics = [key for t in topics for key in t.issue_keys]
    assert sorted(keys_from_topics) == sorted(key_to_topic.keys())
