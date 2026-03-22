from src.mcp.server import _check_env_vars, mcp


def test_server_exists():
    assert mcp is not None
    assert mcp.name == "arcana"


def test_check_env_vars_returns_dict():
    result = _check_env_vars()
    assert isinstance(result, dict)
    assert "SUPABASE_URL" in result
    assert "VOYAGE_API_KEY" in result
    for value in result.values():
        assert value in ("configured", "not_configured")
