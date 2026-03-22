"""Extract corpus .zip archives into working directories.

Looks for .zip files in data/corpus/ and extracts each into a
directory with the same name (e.g., fastmcp.zip → fastmcp/).

Usage:
    python -m scripts.setup_corpus           # extract all, skip existing
    python -m scripts.setup_corpus --force   # re-extract everything
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

CORPUS_ROOT = Path(__file__).parent.parent / "data" / "corpus"


def setup(force: bool = False) -> int:
    zips = sorted(CORPUS_ROOT.glob("*.zip"))

    if not zips:
        print("No .zip files found in data/corpus/")
        return 1

    for zip_path in zips:
        lib_name = zip_path.stem
        target_dir = CORPUS_ROOT / lib_name

        if target_dir.exists() and not force:
            print(f"  {lib_name}: already extracted (use --force to re-extract)")
            continue

        if target_dir.exists():
            shutil.rmtree(target_dir)

        print(f"  {lib_name}: extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

        code_count = len(list(target_dir.rglob("*.py")))
        doc_count = len(list(target_dir.rglob("*.md")) + list(target_dir.rglob("*.mdx")))
        print(f"  {lib_name}: {code_count} Python files, {doc_count} doc files")

    return 0


def main() -> int:
    force = "--force" in sys.argv
    print("Setting up corpus...\n")
    result = setup(force)
    print("\nDone.")
    return result


if __name__ == "__main__":
    sys.exit(main())
