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

# LanceDB's default (non-Tantivy) FTS index only supports a single text
# column per index. Rather than add the tantivy-py dependency just to index
# three columns, we concatenate summary + description + comments into one
# search_text column and index that instead.
FTS_INDEX_COLUMN = "search_text"


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("key", pa.string()),
            pa.field("project_key", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("description", pa.string()),
            pa.field("comments_text", pa.string()),
            pa.field("search_text", pa.string()),
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
                FTS_INDEX_COLUMN, replace=True, base_tokenizer="simple"
            )
        except Exception as exc:
            # Surface this rather than silently degrading to vector-only
            # search: a broken FTS index is easy to miss otherwise.
            import logging

            logging.getLogger(__name__).warning(
                "Could not build FTS index on %s: %s", FTS_INDEX_COLUMN, exc
            )
        if count >= 256:
            try:
                self._table.create_index(vector_column_name="vector", replace=True)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning("Could not build vector index: %s", exc)

    def latest_updated(self, project_key: str) -> str | None:
        """Returns the max 'updated' timestamp already indexed for a project,
        used to drive incremental sync."""
        if self._table.count_rows() == 0:
            return None
        rows = (
            self._table.search()
            .where(f"project_key = '{project_key}'", prefilter=True)
            .select(["updated"])
            .limit(1_000_000)
            .to_list()
        )
        if not rows:
            return None
        values = [r["updated"] for r in rows if r.get("updated")]
        return max(values) if values else None

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

    def has_fts_index(self) -> bool:
        return any(idx.index_type == "FTS" for idx in self._table.list_indices())

    def hybrid_search(
        self, query_text: str, query_vector: list[float], top_k: int, project_key: str | None = None
    ) -> list[dict[str, Any]]:
        if self.has_fts_index():
            query = (
                self._table.search(query_type="hybrid", fts_columns=FTS_INDEX_COLUMN)
                .vector(query_vector)
                .text(query_text)
                .limit(top_k)
            )
            if project_key:
                query = query.where(f"project_key = '{project_key}'", prefilter=True)
            return query.to_list()

        # No FTS index (e.g. ensure_indexes() was never called, or building
        # it failed): fall back to vector-only search rather than erroring.
        vquery = self._table.search(query_vector).limit(top_k)
        if project_key:
            vquery = vquery.where(f"project_key = '{project_key}'", prefilter=True)
        return vquery.to_list()


def issue_to_row(issue: JiraIssue, vector: list[float]) -> dict[str, Any]:
    comments_text = "\n".join(issue.comments)
    search_text = "\n\n".join(p for p in [issue.summary, issue.description, comments_text] if p)
    return {
        "key": issue.key,
        "project_key": issue.project_key,
        "summary": issue.summary,
        "description": issue.description,
        "comments_text": comments_text,
        "search_text": search_text,
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
