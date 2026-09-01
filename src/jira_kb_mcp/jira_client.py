"""Minimal Jira Cloud REST client for indexing purposes.

Uses the current /rest/api/3/search/jql endpoint (token-based pagination),
not the legacy /rest/api/3/search endpoint, which Atlassian has removed.
Read-only: this client never writes anything back to Jira.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import requests

from .config import JiraConfig

FIELDS = [
    "summary",
    "description",
    "status",
    "resolution",
    "issuetype",
    "priority",
    "labels",
    "created",
    "updated",
    "resolutiondate",
    "assignee",
    "reporter",
]

PAGE_SIZE = 100
REQUEST_TIMEOUT = 30


@dataclass
class JiraIssue:
    key: str
    project_key: str
    summary: str
    description: str
    status: str
    resolution: str | None
    issue_type: str
    priority: str | None
    labels: list[str] = field(default_factory=list)
    created: str | None = None
    updated: str | None = None
    resolved: str | None = None
    assignee: str | None = None
    reporter: str | None = None
    comments: list[str] = field(default_factory=list)

    @property
    def url_path(self) -> str:
        return f"/browse/{self.key}"

    def combined_text(self) -> str:
        """Text blob used for embedding + full-text indexing."""
        parts = [self.summary, self.description or ""]
        parts.extend(self.comments)
        return "\n\n".join(p for p in parts if p)


class JiraClientError(RuntimeError):
    pass


class JiraClient:
    def __init__(self, config: JiraConfig, session: requests.Session | None = None):
        self._config = config
        self._session = session or requests.Session()
        self._session.auth = config.auth
        self._session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._config.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 401:
            raise JiraClientError(
                "Jira authentication failed (401). Check JIRA_EMAIL and JIRA_API_TOKEN."
            )
        if resp.status_code == 403:
            raise JiraClientError(
                "Jira access forbidden (403). Check that your account has permission "
                "to browse this project."
            )
        if not resp.ok:
            raise JiraClientError(f"Jira API error {resp.status_code} on {path}: {resp.text[:500]}")
        return resp.json()

    def verify_connection(self) -> str:
        """Returns the display name of the authenticated user, or raises."""
        data = self._get("/rest/api/3/myself")
        return data.get("displayName", "unknown")

    def search_issues(
        self, project_key: str, updated_since: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yields raw issue dicts for a project, paginating via nextPageToken.

        updated_since: optional JQL-style date string (e.g. "2026-01-01") to
        support incremental sync.
        """
        jql = f"project = {project_key} ORDER BY updated ASC"
        if updated_since:
            jql = f'project = {project_key} AND updated >= "{updated_since}" ORDER BY updated ASC'

        next_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "jql": jql,
                "maxResults": PAGE_SIZE,
                "fields": ",".join(FIELDS),
            }
            if next_token:
                params["nextPageToken"] = next_token

            data = self._get("/rest/api/3/search/jql", params=params)
            issues = data.get("issues", [])
            for issue in issues:
                yield issue

            next_token = data.get("nextPageToken")
            if not next_token or not issues:
                break

    def get_comments(self, issue_key: str, max_comments: int = 20) -> list[str]:
        data = self._get(
            f"/rest/api/3/issue/{issue_key}/comment",
            params={"maxResults": max_comments, "orderBy": "-created"},
        )
        texts = []
        for comment in data.get("comments", []):
            texts.append(_adf_to_text(comment.get("body")))
        return [t for t in texts if t]


def _adf_to_text(node: Any) -> str:
    """Extracts plain text from Atlassian Document Format (ADF) content."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        content = node.get("content", [])
        return " ".join(_adf_to_text(child) for child in content)
    if isinstance(node, list):
        return " ".join(_adf_to_text(child) for child in node)
    return ""


def parse_issue(raw: dict[str, Any]) -> JiraIssue:
    fields = raw.get("fields", {})
    key = raw["key"]
    project_key = key.split("-")[0]
    status = (fields.get("status") or {}).get("name", "Unknown")
    resolution = (fields.get("resolution") or {}).get("name")
    issue_type = (fields.get("issuetype") or {}).get("name", "Unknown")
    priority = (fields.get("priority") or {}).get("name")
    assignee = (fields.get("assignee") or {}).get("displayName")
    reporter = (fields.get("reporter") or {}).get("displayName")

    return JiraIssue(
        key=key,
        project_key=project_key,
        summary=fields.get("summary") or "",
        description=_adf_to_text(fields.get("description")),
        status=status,
        resolution=resolution,
        issue_type=issue_type,
        priority=priority,
        labels=fields.get("labels") or [],
        created=fields.get("created"),
        updated=fields.get("updated"),
        resolved=fields.get("resolutiondate"),
        assignee=assignee,
        reporter=reporter,
    )
