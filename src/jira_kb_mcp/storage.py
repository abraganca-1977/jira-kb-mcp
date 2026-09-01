"""LanceDB-backed local storage for indexed Jira issues.

A single embedded table holds structured fields, the full text, and the
embedding vector. This gives us vector search, BM25 full-text search, and
metadata filtering all from one on-disk store, with no server process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from .jira_client import JiraIssue

TABLE_NAME = "issues"
FTS_INDEX_COLUMNS = ["summary", "description", "comments_text"]


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("key", pa.string()),
            pa.field("project_key", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("description", pa.string()),
            pa.field("comments_text", pa.string()),
            pa.field("status", pa.string()),
            pa.field("resolution", pa.string()),
            pa.field("issue_type", pa.string()),
            pa.field("priority", pa.string()),
            pa.field("labels", pa.string()),  # comma-joined for simplicity/FTS
            pa.field("created", pa.string()),
            pa.field("updated", pa.string()),
            pa.field("resolved", pa.string()),
            pa.field("assignee", pa.string()),
            pa.field("reporter", pa.string()),
            pa.field("topic_id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


class Store:
    def __init__(self, data_dir: Path, vector_dim: int):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(data_dir / "lancedb"))
        self._vector_dim = vector_dim
        self._table = self._open_or_create_table()

    def _open_or_create_table(self):
        if TABLE_NAME in self._db.table_names():
            return self._db.open_table(TABLE_NAME)
        return self._db.create_table(TABLE_NAME, schema=_schema(self._vector_dim))

    def upsert_issues(self, rows: list[dict[str, Any]]) -> None:
        """Upsert rows keyed by 'key'. Expects each row to already include a
        'vector' field with the embedding."""
        if not rows:
            return
        (
            self._table.merge_insert("key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def ensure_indexes(self) -> None:
        """Builds/refreshes the vector and full-text indexes.

        Safe to call repeatedly; LanceDB folds in new rows on optimize().
        """
        count = self._table.count_rows()
        if count == 0:
            return
        try:
            self._table.create_fts_index(
                FTS_INDEX_COLUMNS, replace=True, base_tokenizer="simple"
            )
        except Exception:
            # FTS index creation can fail on very small/empty tables; ignore
            # and fall back to flat scan, which LanceDB does automatically.
            pass
        if count >= 256:
            try:
                self._table.create_index(vector_column_name="vector", replace=True)
            except Exception:
                pass

    def latest_updated(self, project_key: str) -> str | None:
        """Returns the max 'updated' timestamp already indexed for a project,
        used to drive incremental sync."""
        if self._table.count_rows() == 0:
            return None
        df = (
            self._table.search()
            .where(f"project_key = '{project_key}'", prefilter=True)
            .select(["updated"])
            .limit(1_000_000)
            .to_pandas()
        )
        if df.empty:
            return None
        return str(df["updated"].max())

    def count(self, project_key: str | None = None) -> int:
        if project_key:
            return len(
                self._table.search()
                .where(f"project_key = '{project_key}'", prefilter=True)
                .select(["key"])
                .limit(1_000_000)
                .to_list()
            )
        return self._table.count_rows()

    def get_by_key(self, key: str) -> dict[str, Any] | None:
        results = self._table.search().where(f"key = '{key}'", prefilter=True).limit(1).to_list()
        return results[0] if results else None

    def all_rows(self, project_key: str | None = None) -> list[dict[str, Any]]:
        query = self._table.search()
        if project_key:
            query = query.where(f"project_key = '{project_key}'", prefilter=True)
        return query.limit(1_000_000).to_list()

    def update_topics(self, key_to_topic: dict[str, int]) -> None:
        rows = [{"key": key, "topic_id": topic} for key, topic in key_to_topic.items()]
        if not rows:
            return
        (
            self._table.merge_insert("key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def hybrid_search(
        self, query_text: str, query_vector: list[float], top_k: int, project_key: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            self._table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
            .limit(top_k)
        )
        if project_key:
            query = query.where(f"project_key = '{project_key}'", prefilter=True)
        try:
            return query.to_list()
        except Exception:
            # No FTS index yet (e.g. empty/small table): fall back to
            # vector-only search.
            vquery = self._table.search(query_vector).limit(top_k)
            if project_key:
                vquery = vquery.where(f"project_key = '{project_key}'", prefilter=True)
            return vquery.to_list()


def issue_to_row(issue: JiraIssue, vector: list[float]) -> dict[str, Any]:
    return {
        "key": issue.key,
        "project_key": issue.project_key,
        "summary": issue.summary,
        "description": issue.description,
        "comments_text": "\n".join(issue.comments),
        "status": issue.status,
        "resolution": issue.resolution or "",
        "issue_type": issue.issue_type,
        "priority": issue.priority or "",
        "labels": ",".join(issue.labels),
        "created": issue.created or "",
        "updated": issue.updated or "",
        "resolved": issue.resolved or "",
        "assignee": issue.assignee or "",
        "reporter": issue.reporter or "",
        "topic_id": -1,
        "vector": vector,
    }
