"""Validate corpus files and report statistics.

Checks that:
- Python files are parseable by tree-sitter
- Markdown/MDX files are non-empty and valid UTF-8
- No binary files are present
- Reports file counts, line counts, and estimated tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

CORPUS_ROOT = Path(__file__).parent.parent / "data" / "corpus"
PY_LANGUAGE = Language(tspython.language())


def _is_binary(path: Path) -> bool:
    """Check if a file appears to be binary."""
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except OSError:
        return True


def validate_python_files(code_dir: Path) -> tuple[int, int, list[str]]:
    """Parse all .py files with tree-sitter, return (count, lines, errors)."""
    parser = Parser(PY_LANGUAGE)
    files = sorted(code_dir.rglob("*.py"))
    total_lines = 0
    errors: list[str] = []

    for f in files:
        if _is_binary(f):
            errors.append(f"BINARY: {f.relative_to(CORPUS_ROOT)}")
            continue
        try:
            source = f.read_bytes()
            tree = parser.parse(source)
            total_lines += source.count(b"\n") + 1
            if tree.root_node.has_error:
                errors.append(f"PARSE_ERROR: {f.relative_to(CORPUS_ROOT)}")
        except Exception as e:
            errors.append(f"READ_ERROR: {f.relative_to(CORPUS_ROOT)}: {e}")

    return len(files), total_lines, errors


def validate_doc_files(docs_dir: Path) -> tuple[int, int, list[str]]:
    """Check all .md/.mdx files are valid UTF-8 and non-empty."""
    files = sorted(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx")))
    total_lines = 0
    errors: list[str] = []

    for f in files:
        if _is_binary(f):
            errors.append(f"BINARY: {f.relative_to(CORPUS_ROOT)}")
            continue
        try:
            content = f.read_text(encoding="utf-8")
            lines = content.count("\n") + 1
            total_lines += lines
            if len(content.strip()) == 0:
                errors.append(f"EMPTY: {f.relative_to(CORPUS_ROOT)}")
        except UnicodeDecodeError:
            errors.append(f"ENCODING_ERROR: {f.relative_to(CORPUS_ROOT)}")
        except Exception as e:
            errors.append(f"READ_ERROR: {f.relative_to(CORPUS_ROOT)}: {e}")

    return len(files), total_lines, errors


def main() -> int:
    libraries = [d for d in CORPUS_ROOT.iterdir() if d.is_dir()]

    if not libraries:
        zips = list(CORPUS_ROOT.glob("*.zip"))
        if zips:
            print("No extracted libraries found. Run setup first:")
            print("  make setup-corpus")
        else:
            print("No libraries found in corpus directory.")
        return 1

    all_errors: list[str] = []

    for lib_dir in sorted(libraries):
        lib_name = lib_dir.name
        print(f"\n{'=' * 60}")
        print(f"Library: {lib_name}")
        print(f"{'=' * 60}")

        code_dir = lib_dir / "code"
        docs_dir = lib_dir / "docs"

        if code_dir.exists():
            py_count, py_lines, py_errors = validate_python_files(code_dir)
            print("\n  Code:")
            print(f"    Python files: {py_count}")
            print(f"    Total lines:  {py_lines:,}")
            print(f"    Parse errors: {len(py_errors)}")
            all_errors.extend(py_errors)
        else:
            print("\n  Code: (no code/ directory)")

        if docs_dir.exists():
            doc_count, doc_lines, doc_errors = validate_doc_files(docs_dir)
            print("\n  Docs:")
            print(f"    Markdown files: {doc_count}")
            print(f"    Total lines:    {doc_lines:,}")
            print(f"    Errors:         {len(doc_errors)}")
            all_errors.extend(doc_errors)
        else:
            print("\n  Docs: (no docs/ directory)")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if all_errors:
        print(f"\n  {len(all_errors)} issue(s) found:\n")
        for err in all_errors:
            print(f"    - {err}")
        return 1
    else:
        print("\n  All files validated successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
