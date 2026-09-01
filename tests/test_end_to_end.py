"""End-to-end test: mocked Jira API -> sync -> local search.

Exercises the full pipeline (jira_client -> embeddings -> storage -> search)
against a fake Jira server via `responses`, so it never touches the network
or requires real credentials.
"""

import responses

from jira_kb_mcp.config import AppConfig, JiraConfig
from jira_kb_mcp.search import search_cases
from jira_kb_mcp.sync import sync_project

BASE_URL = "https://fake.atlassian.net"


def _issue(key: str, summary: str, description: str, status: str = "Done", resolution: str | None = "Fixed"):
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": {
                "type": "doc",
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": description}]}
                ],
            },
            "status": {"name": status},
            "resolution": {"name": resolution} if resolution else None,
            "issuetype": {"name": "Bug"},
            "priority": {"name": "High"},
            "labels": [],
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": "2026-01-02T00:00:00.000+0000",
            "resolutiondate": "2026-01-02T00:00:00.000+0000",
            "assignee": None,
            "reporter": None,
        },
    }


@responses.activate
def test_sync_then_search_finds_relevant_issue(tmp_path):
    issues = [
        _issue(
            "SEC-1",
            "Login times out after 30 seconds",
            "Users report the login page hangs and eventually times out with a 504 error.",
        ),
        _issue(
            "SEC-2",
            "Payment gateway returns declined for valid cards",
            "Cybersource is rejecting transactions with code 05 for cards that should be valid.",
        ),
    ]

    responses.add(
        responses.GET,
        f"{BASE_URL}/rest/api/3/search/jql",
        json={"issues": issues, "nextPageToken": None},
        status=200,
    )
    for issue in issues:
        responses.add(
            responses.GET,
            f"{BASE_URL}/rest/api/3/issue/{issue['key']}/comment",
            json={"comments": []},
            status=200,
        )

    config = AppConfig(
        jira=JiraConfig(base_url=BASE_URL, email="user@example.com", api_token="token"),
        data_dir=tmp_path,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    result = sync_project(config, "SEC", incremental=False)
    assert result.issues_indexed == 2

    hits = search_cases(config, "login page hanging with timeout error", top_k=1)
    assert len(hits) == 1
    assert hits[0].key == "SEC-1"
