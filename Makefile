.PHONY: install test lint format serve index reindex eval

install:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

serve:
	uv run python -m src.mcp.server

# Targets para fases futuras
index:
	uv run python -m scripts.index_library --library $(LIB) --version $(VER) --corpus-path data/corpus/$(LIB)

index-dry:
	uv run python -m scripts.index_library --library $(LIB) --version $(VER) --corpus-path data/corpus/$(LIB) --dry-run

reindex:
	uv run python -m scripts.reindex --library $(LIB)

eval:
	uv run python -m scripts.run_eval --golden-set eval/golden_set.json --output eval/results/

eval-level:
	uv run python -m scripts.run_eval --golden-set eval/golden_set.json --level $(LEVEL) --output eval/results/
