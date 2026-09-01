"""Command-line interface for jira-kb-mcp.

    jira-kb init                 interactive setup, writes ~/.jira-kb-mcp/.env
    jira-kb sync PROJECT         index (or re-index) a Jira project
    jira-kb search "query"       search the local knowledge base
    jira-kb topics               list detected topics
    jira-kb stats PROJECT        show indexing stats for a project
    jira-kb mcp                  run the MCP server (stdio by default)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import ConfigError, DEFAULT_DATA_DIR, load_config

app = typer.Typer(
    name="jira-kb",
    help="Local Jira knowledge base agent: index a project, search past cases, discover topics.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init():
    """Interactive setup: prompts for Jira connection details and saves them
    to ~/.jira-kb-mcp/.env."""
    console.print("[bold]jira-kb-mcp setup[/bold]")
    console.print(
        "You'll need a Jira Cloud API token. Create one at "
        "https://id.atlassian.com/manage-profile/security/api-tokens\n"
    )

    base_url = typer.prompt("Jira URL (e.g. https://yourcompany.atlassian.net)")
    email = typer.prompt("Jira account email")
    api_token = typer.prompt("Jira API token", hide_input=True)

    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    env_path = DEFAULT_DATA_DIR / ".env"
    env_path.write_text(
        f"JIRA_URL={base_url.rstrip('/')}\n"
        f"JIRA_EMAIL={email}\n"
        f"JIRA_API_TOKEN={api_token}\n"
    )
    env_path.chmod(0o600)
    console.print(f"\n[green]Saved configuration to {env_path}[/green]")
    console.print("Run [bold]jira-kb sync PROJECT_KEY[/bold] to index a project.")


@app.command()
def sync(
    project_key: str = typer.Argument(..., help="Jira project key, e.g. SEC or PD"),
    full: bool = typer.Option(False, help="Force a full re-index instead of incremental sync."),
    no_comments: bool = typer.Option(False, help="Skip fetching issue comments (faster, less context)."),
):
    """Indexes a Jira project into the local knowledge base."""
    from .sync import sync_project

    config = _load_config_or_exit()

    with console.status(f"Syncing project {project_key}..."):
        def _progress(count: int, key: str):
            if count % 10 == 0:
                console.print(f"  ...{count} issues processed (last: {key})")

        result = sync_project(
            config,
            project_key,
            incremental=not full,
            fetch_comments=not no_comments,
            progress_callback=_progress,
        )

    console.print(
        f"[green]Done.[/green] Indexed {result.issues_indexed} issues, "
        f"detected {result.topics_detected} topics for project {result.project_key}."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Description of the problem to search for."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to show."),
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict search to a project key."),
):
    """Searches the local knowledge base for similar past Jira issues."""
    from .search import search_cases

    config = _load_config_or_exit()
    hits = search_cases(config, query, top_k=top_k, project_key=project)

    if not hits:
        console.print("[yellow]No matches found. Has this project been synced? (jira-kb sync PROJECT)[/yellow]")
        return

    table = Table(show_lines=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Summary")
    table.add_column("Status")
    table.add_column("Resolution")
    for hit in hits:
        table.add_row(hit.key, hit.summary, hit.status, hit.resolution or "-")
    console.print(table)


@app.command()
def topics(
    project: str | None = typer.Option(None, "--project", "-p", help="Restrict to a project key."),
):
    """Lists recurring topics detected across indexed issues."""
    from .search import list_topics

    config = _load_config_or_exit()
    result = list_topics(config, project_key=project)

    if not result:
        console.print(
            "[yellow]No topics detected yet (need at least a few dozen indexed issues).[/yellow]"
        )
        return

    table = Table(show_lines=True)
    table.add_column("Topic", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Sample issues")
    for t in result:
        table.add_row(t["label"], str(t["size"]), ", ".join(t["issue_keys"][:5]))
    console.print(table)


@app.command()
def stats(project_key: str = typer.Argument(..., help="Jira project key.")):
    """Shows how many issues are indexed for a project."""
    from .search import project_stats

    config = _load_config_or_exit()
    result = project_stats(config, project_key)
    console.print(result)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", help="stdio (default) or streamable-http."),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    """Runs the MCP server so an AI agent (Claude, ChatGPT, Kiro) can use
    this knowledge base directly in a conversation."""
    from .mcp_server import mcp as mcp_instance

    _load_config_or_exit()
    if transport == "streamable-http":
        mcp_instance.run(transport="streamable-http", host=host, port=port)
    else:
        mcp_instance.run()


def _load_config_or_exit():
    try:
        return load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
