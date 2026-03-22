import os

from loguru import logger

_client = None


def get_client():
    """Get the Supabase client singleton. Returns None if not configured."""
    global _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key or url.startswith("https://your-"):
        return None

    if _client is None:
        from supabase import create_client

        _client = create_client(url, key)
        logger.bind(operation="db_connect").info("Supabase client initialized")

    return _client


def check_connection() -> dict:
    """Check Supabase connectivity. Returns status dict for health check."""
    client = get_client()

    if client is None:
        return {"status": "not_configured", "message": "SUPABASE_URL or SUPABASE_KEY not set"}

    try:
        # Simple query to verify connectivity
        client.table("source_files").select("id").limit(1).execute()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
