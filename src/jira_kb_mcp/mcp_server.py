"""MCP server exposing the local Jira knowledge base as tools.

Works with any MCP-capable host (Claude Desktop/Code, Kiro, ChatGPT via
Developer Mode). Default transport is stdio; pass --transport streamable-http
to serve over HTTP instead (see README for exposing it to ChatGPT).
"""

from __future__ import annotations

from mcp.server import MCPServer

from .config import ConfigError, load_config
from .search import get_case_detail, list_topics, project_stats, search_cases
from .sync import sync_project

mcp = MCPServer(
    "jira-kb",
    instructions=(
        "Provides access to a local knowledge base indexed from Jira issues. "
        "Use search_cases to find past issues similar to a problem description "
        "before proposing a new solution. Use list_topics to see recurring "
        "themes across the indexed project. Call sync_jira_project first if "
        "the project has not been indexed yet."
    ),
)


@mcp.tool()
def sync_jira_project(project_key: str, incremental: bool = True) -> dict:
    """Indexes (or re-indexes) a Jira project into the local knowledge base.

    Args:
        project_key: The Jira project key, e.g. "SEC" or "PD".
        incremental: If True (default), only fetches issues updated since the
            last sync. Set False to force a full re-index.
    """
    config = load_config()
    result = sync_project(config, project_key, incremental=incremental)
    return {
        "project_key": result.project_key,
        "issues_indexed": result.issues_indexed,
        "topics_detected": result.topics_detected,
    }


@mcp.tool()
def search_cases_tool(query: str, top_k: int = 5, project_key: str | None = None) -> list[dict]:
    """Searches the local knowledge base for past Jira issues similar to a
    problem description, combining semantic and keyword matching.

    Args:
        query: A description of the problem or symptom to search for.
        top_k: Maximum number of results to return (default 5).
        project_key: Optional Jira project key to restrict the search to.
    """
    config = load_config()
    hits = search_cases(config, query, top_k=top_k, project_key=project_key)
    return [
        {
            "key": h.key,
            "summary": h.summary,
            "status": h.status,
            "resolution": h.resolution,
            "issue_type": h.issue_type,
            "updated": h.updated,
            "url": h.url,
        }
        for h in hits
    ]


@mcp.tool()
def get_case(issue_key: str) -> dict:
    """Returns the full stored detail for a single indexed Jira issue.

    Args:
        issue_key: The Jira issue key, e.g. "SEC-1470".
    """
    config = load_config()
    detail = get_case_detail(config, issue_key)
    if detail is None:
        return {"error": f"Issue {issue_key} not found in the local knowledge base."}
    return detail


@mcp.tool()
def list_topics_tool(project_key: str | None = None) -> list[dict]:
    """Lists recurring topics/themes detected across indexed Jira issues,
    each with a readable label and its most representative issue keys.

    Args:
        project_key: Optional Jira project key to restrict the topic list to.
    """
    config = load_config()
    return list_topics(config, project_key=project_key)


@mcp.tool()
def get_project_stats(project_key: str) -> dict:
    """Returns how many issues are indexed for a project and the last sync
    watermark, useful to check whether sync_jira_project needs to run.

    Args:
        project_key: The Jira project key, e.g. "SEC" or "PD".
    """
    config = load_config()
    return project_stats(config, project_key)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="jira-kb-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport to serve over. stdio (default) works with Claude "
        "Desktop/Code and Kiro. streamable-http opens an HTTP port, needed "
        "for ChatGPT Developer Mode connectors.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        load_config()
    except ConfigError as exc:
        # Fail loudly at startup rather than on the first tool call, so the
        # host surfaces a clear error instead of a silent tool failure.
        raise SystemExit(str(exc))

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
