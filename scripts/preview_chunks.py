"""Preview code chunks generated from Python files.

Usage:
    uv run python -m scripts.preview_chunks FILE_OR_DIR [--verbose] [--stats]

Examples:
    # Single file
    uv run python -m scripts.preview_chunks data/corpus/fastmcp/code/tools/function_tool.py

    # Directory (all .py files)
    uv run python -m scripts.preview_chunks data/corpus/fastmcp/code/tools/

    # Verbose: include signature and docstring
    uv run python -m scripts.preview_chunks data/corpus/fastmcp/code/tools/function_tool.py -v

    # Stats only: aggregate statistics
    uv run python -m scripts.preview_chunks data/corpus/fastmcp/code/ --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken

from src.indexing.code_chunker import CodeChunk, chunk_file

CORPUS_ROOT = Path(__file__).parent.parent / "data" / "corpus"

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _relative_filepath(path: Path) -> str:
    """Derive a relative filepath suitable for chunk_file."""
    # Try to make it relative to corpus/{lib}/code/
    for parent in path.parents:
        if parent.name == "code" and parent.parent.parent == CORPUS_ROOT:
            return str(path.relative_to(parent))
    return path.name


def _print_file_chunks(filepath: Path, chunks: list[CodeChunk], verbose: bool) -> None:
    rel = _relative_filepath(filepath)
    print(f"\n{'=' * 70}")
    print(f"  {rel}")
    print(f"{'=' * 70}")

    for c in chunks:
        tokens = _count_tokens(c.source_code)
        token_str = f"{tokens:>6,} tok"
        line_range = f"[{c.start_line}:{c.end_line}]"
        print(f"  {c.chunk_type:8s} {c.symbol_name:45s} {line_range:>12s}  {token_str}")
        if verbose:
            if c.signature:
                sig_first = c.signature.split("\n")[0]
                print(f"           sig: {sig_first}")
            if c.docstring:
                doc_first = c.docstring.split("\n")[0][:70]
                print(f"           doc: {doc_first}")
            if c.imports:
                print(f"           imports: {len(c.imports)}")


def _print_stats(all_chunks: list[CodeChunk], file_count: int) -> None:
    counts = {"module": 0, "class": 0, "method": 0, "function": 0}
    token_sizes: list[int] = []

    for c in all_chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
        token_sizes.append(_count_tokens(c.source_code))

    print(f"\n{'=' * 70}")
    print("  STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Files processed: {file_count}")
    print(f"  Total chunks:    {len(all_chunks)}")
    print(
        f"    module:   {counts['module']:>5}"
        f"    class: {counts['class']:>5}"
        f"    method: {counts['method']:>5}"
        f"    function: {counts['function']:>5}"
    )

    if token_sizes:
        token_sizes.sort()
        median = token_sizes[len(token_sizes) // 2]
        mean = sum(token_sizes) // len(token_sizes)
        over_800 = sum(1 for t in token_sizes if t > 800)
        over_16k = sum(1 for t in token_sizes if t > 16384)

        print("\n  Token distribution:")
        print(
            f"    min: {token_sizes[0]:,}"
            f"    median: {median:,}"
            f"    mean: {mean:,}"
            f"    max: {token_sizes[-1]:,}"
        )
        print(f"    >800 tokens:  {over_800:>5} chunks  ({over_800 * 100 / len(token_sizes):.1f}%)")
        print(f"    >16K tokens:  {over_16k:>5} chunks  ({over_16k * 100 / len(token_sizes):.1f}%)")

        if over_16k > 0:
            print("\n  Chunks exceeding 16K tokens (Voyage Code 3 limit):")
            large = [
                (c, _count_tokens(c.source_code))
                for c in all_chunks
                if _count_tokens(c.source_code) > 16384
            ]
            large.sort(key=lambda x: -x[1])
            for c, t in large[:10]:
                print(f"    {c.symbol_name} ({c.chunk_type}) — {t:,} tokens")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview code chunks from Python files")
    parser.add_argument(
        "path",
        type=Path,
        help="Python file or directory to chunk",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show signature and docstring for each chunk",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregate statistics only",
    )
    args = parser.parse_args()

    target: Path = args.path
    if not target.exists():
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 1

    if target.is_file():
        py_files = [target]
    else:
        py_files = sorted(target.rglob("*.py"))

    if not py_files:
        print(f"No .py files found in {target}", file=sys.stderr)
        return 1

    all_chunks: list[CodeChunk] = []
    errors: list[str] = []

    for f in py_files:
        try:
            source = f.read_bytes()
            rel = _relative_filepath(f)
            chunks = chunk_file(source, rel)
            all_chunks.extend(chunks)

            if not args.stats:
                _print_file_chunks(f, chunks, args.verbose)
        except Exception as e:
            errors.append(f"{f}: {e}")

    if args.stats or len(py_files) > 1:
        _print_stats(all_chunks, len(py_files))

    if errors:
        print(f"\n  {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
