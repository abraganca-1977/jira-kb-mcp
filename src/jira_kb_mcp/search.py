"""Search and topic-listing helpers shared by the CLI and the MCP server."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .embeddings import get_embedder
from .storage import Store
from .topics import compute_topics


@dataclass
class SearchHit:
    key: str
    summary: str
    status: str
    resolution: str
    issue_type: str
    updated: str
    url: str


def _row_to_hit(row: dict, base_url: str) -> SearchHit:
    return SearchHit(
        key=row["key"],
        summary=row["summary"],
        status=row["status"],
        resolution=row.get("resolution") or "",
        issue_type=row.get("issue_type", ""),
        updated=row.get("updated", ""),
        url=f"{base_url}/browse/{row['key']}",
    )


def search_cases(config: AppConfig, query: str, top_k: int = 5, project_key: str | None = None) -> list[SearchHit]:
    embedder = get_embedder(config.embedding_model)
    store = Store(config.data_dir, vector_dim=embedder.dimension)
    query_vector = embedder.embed_query(query)
    rows = store.hybrid_search(query, query_vector, top_k=top_k, project_key=project_key)
    return [_row_to_hit(r, config.jira.base_url) for r in rows]


def get_case_detail(config: AppConfig, issue_key: str) -> dict | None:
    embedder = get_embedder(config.embedding_model)
    store = Store(config.data_dir, vector_dim=embedder.dimension)
    row = store.get_by_key(issue_key)
    if not row:
        return None
    row["url"] = f"{config.jira.base_url}/browse/{issue_key}"
    row.pop("vector", None)
    return row


def list_topics(config: AppConfig, project_key: str | None = None) -> list[dict]:
    embedder = get_embedder(config.embedding_model)
    store = Store(config.data_dir, vector_dim=embedder.dimension)
    rows = store.all_rows(project_key=project_key)
    _, topics = compute_topics(rows)
    return [
        {
            "topic_id": t.topic_id,
            "label": t.label,
            "size": t.size,
            "issue_keys": t.issue_keys[:10],
        }
        for t in topics
    ]


def project_stats(config: AppConfig, project_key: str) -> dict:
    embedder = get_embedder(config.embedding_model)
    store = Store(config.data_dir, vector_dim=embedder.dimension)
    return {
        "project_key": project_key,
        "issues_indexed": store.count(project_key),
        "last_updated_seen": store.latest_updated(project_key),
    }
