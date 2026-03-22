# Arcana

Experimental MCP server for internal technical knowledge. Implements three retrieval architectures (pure retrieval, retrieval + synthesis, agentic retrieval) to compare RAG approaches over the same indexed data.

**Stack**: Python + FastMCP | Supabase (PostgreSQL + pgvector) | Voyage AI | Anthropic/OpenAI | Loguru

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Supabase account (free tier)

## Environment Variables

The following variables must be defined in `.devcontainer/.env` before starting the container:

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL (Settings > API) |
| `SUPABASE_KEY` | Yes | Supabase anon key (Settings > API) |
| `VOYAGE_API_KEY` | Yes | Voyage AI API key for embeddings |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (for synthesis levels 2 & 3) |
| `OPENAI_API_KEY` | No | OpenAI API key (alternative for synthesis) |

## Setup

### 1. Set up Supabase

1. Create a project at [supabase.com](https://supabase.com) (free tier works)
2. Copy your **Project URL** and **anon key** from Settings > API
3. Go to **SQL Editor** in the Supabase dashboard
4. Paste and run the contents of `sql/schema.sql`
5. Verify the tables (`source_files`, `code_chunks`, `doc_chunks`) appear in Table Editor

### 2. Start the DevContainer

This project uses a devcontainer. Open in VS Code with the Dev Containers extension or GitHub Codespaces — dependencies install automatically on container creation.

### 3. Run the server

```bash
make serve
```

## Commands

```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Check linting and formatting
make format     # Auto-format code
make serve      # Start MCP server
```

## Project Structure

```
src/
├── mcp/            # FastMCP server and tools
├── db/             # Supabase client and queries
├── indexing/       # AST-aware chunking and embeddings (Phase 2)
├── retrieval/      # Vector, full-text, and hybrid search (Phase 3+)
├── synthesis/      # Server-side LLM generation (Phase 4+)
└── logger.py       # Structured JSONL logging (loguru)
sql/                # Database schema
scripts/            # CLI tools for indexing and evaluation
data/corpus/        # Source material (code and docs)
eval/               # Evaluation framework
tests/              # Unit tests
```