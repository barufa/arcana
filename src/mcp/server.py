import os

from fastmcp import FastMCP

from src.db.client import check_connection
from src.logger import log_operation

mcp = FastMCP(
    name="arcana",
    instructions="Internal knowledge MCP server for technical library documentation and code.",
    version="0.1.0",
)


def _check_env_vars() -> dict[str, str]:
    """Check which environment variables are configured."""
    vars_to_check = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY", ""),
        "VOYAGE_API_KEY": os.getenv("VOYAGE_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    }
    return {
        name: "configured" if val and not val.startswith("your-") else "not_configured"
        for name, val in vars_to_check.items()
    }


@mcp.tool
def health_check() -> dict:
    """Check server health, environment configuration, and database connectivity."""
    with log_operation("health_check"):
        env_status = _check_env_vars()
        db_status = check_connection()

        all_ok = db_status.get("status") == "connected"

        return {
            "status": "healthy" if all_ok else "degraded",
            "version": "0.1.0",
            "services": {
                "supabase": db_status,
            },
            "environment": env_status,
        }


if __name__ == "__main__":
    mcp.run()
