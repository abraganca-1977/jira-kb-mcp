from jira_kb_mcp.jira_client import _adf_to_text, parse_issue


def test_adf_to_text_plain_string():
    assert _adf_to_text("hello") == "hello"


def test_adf_to_text_none():
    assert _adf_to_text(None) == ""


def test_adf_to_text_nested_document():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Login fails"},
                    {"type": "text", "text": "with a 500 error."},
                ],
            }
        ],
    }
    result = _adf_to_text(doc)
    assert "Login fails" in result
    assert "500 error" in result


def test_parse_issue_basic():
    raw = {
        "key": "SEC-42",
        "fields": {
            "summary": "Login times out",
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Users report timeouts."}],
                    }
                ],
            },
            "status": {"name": "Done"},
            "resolution": {"name": "Fixed"},
            "issuetype": {"name": "Bug"},
            "priority": {"name": "High"},
            "labels": ["auth", "timeout"],
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": "2026-01-05T00:00:00.000+0000",
            "resolutiondate": "2026-01-05T00:00:00.000+0000",
            "assignee": {"displayName": "Ana"},
            "reporter": {"displayName": "Beto"},
        },
    }
    issue = parse_issue(raw)
    assert issue.key == "SEC-42"
    assert issue.project_key == "SEC"
    assert issue.summary == "Login times out"
    assert "timeouts" in issue.description
    assert issue.status == "Done"
    assert issue.resolution == "Fixed"
    assert issue.issue_type == "Bug"
    assert issue.priority == "High"
    assert issue.labels == ["auth", "timeout"]
    assert issue.assignee == "Ana"
    assert issue.reporter == "Beto"


def test_parse_issue_missing_optional_fields():
    raw = {
        "key": "PD-1",
        "fields": {
            "summary": "Something",
            "status": {"name": "Open"},
            "issuetype": {"name": "Task"},
        },
    }
    issue = parse_issue(raw)
    assert issue.resolution is None
    assert issue.priority is None
    assert issue.assignee is None
    assert issue.labels == []
    assert issue.description == ""


def test_combined_text_includes_comments():
    from jira_kb_mcp.jira_client import JiraIssue

    issue = JiraIssue(
        key="SEC-1",
        project_key="SEC",
        summary="Summary line",
        description="Description line",
        status="Open",
        resolution=None,
        issue_type="Bug",
        priority=None,
        comments=["First comment", "Second comment"],
    )
    text = issue.combined_text()
    assert "Summary line" in text
    assert "Description line" in text
    assert "First comment" in text
    assert "Second comment" in text
