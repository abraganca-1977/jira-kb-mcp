# jira-kb-mcp

A local knowledge base built from your Jira project's history, exposed as an
[MCP](https://modelcontextprotocol.io) server so any MCP-capable AI assistant
(Claude Desktop, Claude Code, Kiro, ChatGPT in Developer Mode) can search past
issues and their resolutions while helping you troubleshoot a new one.

Everything runs locally. No database server, no cloud account, no API key
required by default:

- **Storage**: [LanceDB](https://lancedb.com), an embedded vector database.
  One local directory, no server process.
- **Embeddings**: [fastembed](https://github.com/qdrant/fastembed) running a
  multilingual model on ONNX Runtime, fully offline, no GPU needed.
- **Search**: hybrid (semantic + BM25 keyword) search over your indexed
  issues, so both "something like this happened before" and "the exact error
  code" queries work.
- **Topics**: issues are clustered so you get a browsable index of recurring
  themes, not just a flat list of tickets.

## How it works

1. `jira-kb init` — one-time setup, stores your Jira URL/email/API token in
   `~/.jira-kb-mcp/.env` (never committed, file permissions locked to your user).
2. `jira-kb sync PROJECT_KEY` — pulls every issue (summary, description,
   comments, resolution) from a Jira project via the REST API, embeds it, and
   stores it locally. Safe to re-run: subsequent syncs are incremental.
3. Point your AI assistant at the MCP server (see below) and ask it things
   like "have we seen a timeout like this before in PROJ?" — it will call
   `search_cases` and ground its answer in your real ticket history.

You can also use it as a plain CLI without any AI assistant involved:
`jira-kb search "some description of the problem"`.

## Requirements

- Python 3.10+
- A Jira Cloud site and an [API token](https://id.atlassian.com/manage-profile/security/api-tokens)
  for your account (Jira Server/Data Center is not supported yet — see
  Limitations below)

## Install

```bash
git clone <this-repo-url> jira-kb-mcp
cd jira-kb-mcp
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Setup

```bash
jira-kb init
```

This prompts for your Jira URL, account email, and API token, and saves them
to `~/.jira-kb-mcp/.env`. You can also copy `env.example` to `.env` yourself
and fill it in manually — either in the current directory or in
`~/.jira-kb-mcp/.env`.

## Index a project

```bash
jira-kb sync PROJ            # incremental sync (default)
jira-kb sync PROJ --full     # force full re-index
```

This fetches every issue in the project (paginated, using Jira's current
`/rest/api/3/search/jql` endpoint), embeds the summary + description +
comments, and stores it in `~/.jira-kb-mcp/lancedb/`. Re-running `sync` only
fetches issues updated since the last run.

## Use it from the CLI

```bash
jira-kb search "connection timeout when calling payment gateway"
jira-kb topics
jira-kb stats PROJ
```

## Use it as an MCP agent

The server exposes these tools: `sync_jira_project`, `search_cases_tool`,
`get_case`, `list_topics_tool`, `get_project_stats`.

### Claude Desktop / Claude Code / Kiro (stdio)

Add to your MCP config (`claude_desktop_config.json`, `.mcp.json`, or Kiro's
`.kiro/settings/mcp.json`):

```json
{
  "mcpServers": {
    "jira-kb": {
      "command": "/absolute/path/to/jira-kb-mcp/.venv/bin/jira-kb",
      "args": ["mcp"]
    }
  }
}
```

Restart the assistant, and it will be able to call the tools above during a
conversation.

### ChatGPT (Developer Mode, streamable-http)

ChatGPT's MCP support requires an HTTP(S) endpoint, it cannot launch a local
stdio process the way Claude/Kiro do. Run the server over HTTP:

```bash
jira-kb mcp --transport streamable-http --port 8000
```

This serves `http://127.0.0.1:8000/mcp`. To connect it from ChatGPT you need
that URL to be reachable from OpenAI's servers, which means either:

- Deploying the server somewhere with a public HTTPS URL, or
- Using a tunnel (e.g. `ngrok http 8000`) for local testing.

**Security note**: exposing this server to the internet exposes your Jira
issue data (via the tools) to anyone who can reach that URL. This project
does not implement authentication on the HTTP transport. If you expose it
beyond your own machine, put it behind your own auth (reverse proxy, VPN, or
a tunnel provider's access controls) — do not expose it publicly unauthenticated.

## Configuration reference

All settings are environment variables (see `env.example`):

| Variable | Required | Default | Description |
|---|---|---|---|
| `JIRA_URL` | yes | — | Your Jira Cloud site URL |
| `JIRA_EMAIL` | yes | — | Account email for the API token |
| `JIRA_API_TOKEN` | yes | — | Jira API token |
| `JIRA_KB_DATA_DIR` | no | `~/.jira-kb-mcp` | Where the local LanceDB store lives |
| `JIRA_KB_EMBEDDING_MODEL` | no | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Any [fastembed-supported model](https://qdrant.github.io/fastembed/examples/Supported_Models/) |

## Limitations

- Jira Cloud only (uses the current `/rest/api/3/search/jql` REST endpoint).
  Jira Server/Data Center support is not implemented.
- Topic detection is unsupervised clustering with TF-IDF labels, not an LLM
  summary — labels are keyword lists, not full sentences. It also needs a
  minimum number of indexed issues (5) to produce any clusters.
- Read-only: this tool never writes back to Jira.
- The streamable-http transport has no built-in authentication (see security
  note above).

## License

MIT — see [LICENSE](LICENSE).
