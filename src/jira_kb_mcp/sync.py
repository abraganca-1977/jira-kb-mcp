"""Orchestrates indexing a Jira project into the local knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig
from .embeddings import get_embedder
from .jira_client import JiraClient, parse_issue
from .storage import Store, issue_to_row
from .topics import compute_topics

COMMENT_FETCH_BATCH_LOG_EVERY = 25


@dataclass
class SyncResult:
    project_key: str
    issues_indexed: int
    topics_detected: int


def sync_project(
    config: AppConfig,
    project_key: str,
    incremental: bool = True,
    fetch_comments: bool = True,
    progress_callback=None,
) -> SyncResult:
    """Fetches issues for a project from Jira, embeds them, and stores them
    locally. Returns a summary of what was indexed.

    progress_callback, if given, is called as progress_callback(count, key)
    after each issue is processed, useful for CLI/MCP progress reporting.
    """
    jira = JiraClient(config.jira)
    embedder = get_embedder(config.embedding_model)
    store = Store(config.data_dir, vector_dim=embedder.dimension)

    updated_since = store.latest_updated(project_key) if incremental else None

    rows_batch = []
    count = 0
    for raw in jira.search_issues(project_key, updated_since=updated_since):
        issue = parse_issue(raw)
        if fetch_comments:
            try:
                issue.comments = jira.get_comments(issue.key)
            except Exception:
                issue.comments = []

        vector = embedder.embed_documents([issue.combined_text()])[0]
        rows_batch.append(issue_to_row(issue, vector))
        count += 1

        if progress_callback:
            progress_callback(count, issue.key)

        if len(rows_batch) >= 50:
            store.upsert_issues(rows_batch)
            rows_batch = []

    if rows_batch:
        store.upsert_issues(rows_batch)

    store.ensure_indexes()

    all_rows = store.all_rows(project_key=project_key)
    key_to_topic, topics = compute_topics(all_rows)
    if key_to_topic:
        store.update_topics(key_to_topic)

    return SyncResult(
        project_key=project_key,
        issues_indexed=count,
        topics_detected=len(topics),
    )
