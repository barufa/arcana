"""Tests for the semantic markdown/MDX documentation chunker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.indexing.markdown_chunker import (
    DocChunk,
    _build_header_path,
    _build_section_tree,
    _chunk_changelog,
    _content_hash,
    _extract_code_blocks_from_text,
    _extract_last_sentences,
    _parse_changelog_updates,
    _Section,
    _strip_frontmatter,
    _strip_mdx_components,
    _strip_mdx_imports,
    chunk_file,
)

CORPUS_DOCS = Path(__file__).parent.parent / "data" / "corpus" / "fastmcp" / "docs"

# Use a simple word-based counter for all tests to avoid needing Voyage API key
_mock_counter = lambda text: max(1, int(len(text.split()) * 1.3)) if text else 0  # noqa: E731


def _chunk_file_mocked(source: bytes, filepath: str, **kwargs):
    """Call chunk_file with mocked token counter."""
    with patch("src.indexing.markdown_chunker._count_tokens", side_effect=_mock_counter):
        return chunk_file(source, filepath, **kwargs)


# ---------------------------------------------------------------------------
# Unit tests: string helpers
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_different_input(self):
        assert _content_hash("hello") != _content_hash("world")

    def test_non_empty(self):
        assert len(_content_hash("test")) == 64  # SHA-256 hex


class TestStripFrontmatter:
    def test_with_frontmatter(self):
        text = "---\ntitle: Test\nicon: star\n---\n\n# Hello"
        meta, clean = _strip_frontmatter(text)
        assert meta is not None
        assert meta["title"] == "Test"
        assert "# Hello" in clean
        assert "---" not in clean

    def test_without_frontmatter(self):
        text = "# Hello\n\nSome text."
        meta, clean = _strip_frontmatter(text)
        assert meta is None
        assert clean == text

    def test_invalid_yaml(self):
        text = "---\n: : : invalid\n---\n\nContent"
        meta, clean = _strip_frontmatter(text)
        assert meta is None
        assert "Content" in clean


class TestStripMdxImports:
    def test_removes_imports(self):
        text = "import { VersionBadge } from '/snippets/version-badge.mdx'\n\n# Title"
        result = _strip_mdx_imports(text)
        assert "import" not in result
        assert "# Title" in result

    def test_preserves_non_imports(self):
        text = "# Title\n\nSome text about imports."
        assert _strip_mdx_imports(text) == text


class TestStripMdxComponents:
    def test_tip(self):
        text = "<Tip>Use this carefully.</Tip>"
        result = _strip_mdx_components(text)
        assert "> **Tip:** Use this carefully." in result

    def test_warning(self):
        text = "<Warning>Danger ahead!</Warning>"
        result = _strip_mdx_components(text)
        assert "> **Warning:** Danger ahead!" in result

    def test_version_badge(self):
        text = '<VersionBadge version="2.0.0" />'
        result = _strip_mdx_components(text)
        assert "[Since v2.0.0]" in result

    def test_card(self):
        text = '<Card title="My Card">Card content here.</Card>'
        result = _strip_mdx_components(text)
        assert "#### My Card" in result
        assert "Card content here." in result

    def test_codegroup_unwrapped(self):
        text = "<CodeGroup>\n```python\nprint('hi')\n```\n</CodeGroup>"
        result = _strip_mdx_components(text)
        assert "<CodeGroup>" not in result
        assert "print('hi')" in result

    def test_param_field(self):
        text = '<ParamField name="timeout" type="int">Seconds to wait.</ParamField>'
        result = _strip_mdx_components(text)
        assert "**timeout**" in result
        assert "(*int*)" in result


# ---------------------------------------------------------------------------
# Unit tests: changelog
# ---------------------------------------------------------------------------


class TestParseChangelogUpdates:
    def test_extracts_updates(self):
        text = """
<Update label="v1.0.0" description="2026-01-01">
First release content.
</Update>

<Update label="v1.1.0" description="2026-02-01">
Second release.
</Update>
"""
        updates = _parse_changelog_updates(text)
        assert len(updates) == 2
        assert updates[0][0] == "v1.0.0"
        assert updates[0][1] == "2026-01-01"
        assert "First release" in updates[0][2]
        assert updates[1][0] == "v1.1.0"

    def test_no_updates(self):
        assert _parse_changelog_updates("# Just a heading") == []


class TestChunkChangelog:
    def test_produces_changelog_chunks(self):
        text = """---
title: "Changelog"
---

<Update label="v2.0.0" description="2026-03-15">

### Features
- New API endpoint
- Better logging

</Update>

<Update label="v1.0.0" description="2026-01-01">

Initial release.

</Update>
"""
        source_lines = text.split("\n")
        chunks = _chunk_changelog(text, "changelog.mdx", {"title": "Changelog"}, source_lines)
        assert len(chunks) == 2
        assert all(c.content_type == "changelog" for c in chunks)
        assert chunks[0].header_path == "Changelog > v2.0.0"
        assert chunks[1].header_path == "Changelog > v1.0.0"
        assert "New API endpoint" in chunks[0].content

    def test_each_version_separate(self):
        text = """<Update label="v3.0" description="2026-03-01">Content A</Update>
<Update label="v2.0" description="2026-02-01">Content B</Update>
<Update label="v1.0" description="2026-01-01">Content C</Update>"""
        chunks = _chunk_changelog(text, "cl.mdx", None, text.split("\n"))
        assert len(chunks) == 3
        labels = [c.header_path.split(" > ")[-1] for c in chunks]
        assert labels == ["v3.0", "v2.0", "v1.0"]


# ---------------------------------------------------------------------------
# Unit tests: code block extraction
# ---------------------------------------------------------------------------


class TestCodeBlockExtraction:
    def test_extracts_fenced_blocks(self):
        text = "Some text\n\n```python\nprint('hello')\n```\n\nMore text\n\n```bash\necho hi\n```"
        blocks = _extract_code_blocks_from_text(text)
        assert len(blocks) == 2
        assert "print('hello')" in blocks[0]
        assert "echo hi" in blocks[1]

    def test_no_code_blocks(self):
        assert _extract_code_blocks_from_text("Just plain text.") == []


# ---------------------------------------------------------------------------
# Unit tests: section tree
# ---------------------------------------------------------------------------


class TestBuildSectionTree:
    def _parse(self, md_text: str):
        import mistune

        md = mistune.create_markdown(renderer="ast", plugins=["table"])
        tokens = md(md_text)
        lines = md_text.split("\n")
        return _build_section_tree(tokens, lines)

    def test_simple_headings(self):
        md = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        tree = self._parse(md)
        assert tree.level == 0
        assert len(tree.children) == 1  # h1
        h1 = tree.children[0]
        assert h1.title == "Title"
        assert len(h1.children) == 2  # h2 A and h2 B

    def test_deep_nesting(self):
        md = "# L1\n\n## L2\n\n### L3\n\n#### L4\n\n##### L5\n\nDeep content."
        tree = self._parse(md)
        h1 = tree.children[0]
        assert h1.title == "L1"
        h2 = h1.children[0]
        assert h2.title == "L2"
        h3 = h2.children[0]
        assert h3.title == "L3"

    def test_out_of_order_headings(self):
        """h1 then h3 (skipping h2) should not crash."""
        md = "# Title\n\n### Subsection\n\nContent."
        tree = self._parse(md)
        h1 = tree.children[0]
        assert h1.title == "Title"
        # h3 becomes child of h1 since there's no h2
        assert len(h1.children) == 1
        assert h1.children[0].title == "Subsection"

    def test_no_headings(self):
        tree = self._parse("Just some text without any headings.\n\nAnother paragraph.")
        assert tree.level == 0
        assert len(tree.children) == 0
        assert len(tree.tokens) > 0  # content goes to root


class TestBuildHeaderPath:
    def test_nested(self):
        root = _Section(level=0, title="(root)")
        h1 = _Section(level=1, title="Guide", parent=root)
        h2 = _Section(level=2, title="Setup", parent=h1)
        path = _build_header_path(h2, "My Doc")
        assert path == "My Doc > Guide > Setup"

    def test_root_only(self):
        root = _Section(level=0, title="(root)")
        path = _build_header_path(root, "Title")
        assert path == "Title"

    def test_no_title(self):
        root = _Section(level=0, title="(root)")
        h1 = _Section(level=1, title="Section", parent=root)
        path = _build_header_path(h1, None)
        assert path == "Section"


# ---------------------------------------------------------------------------
# Unit tests: overlap
# ---------------------------------------------------------------------------


class TestExtractLastSentences:
    def test_extracts_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _extract_last_sentences(text, 2)
        assert result is not None
        assert "Third sentence." in result
        assert "Fourth sentence." in result

    def test_single_sentence(self):
        assert _extract_last_sentences("Only one sentence here") is None

    def test_empty(self):
        assert _extract_last_sentences("") is None


# ---------------------------------------------------------------------------
# Unit tests: chunk_file (mocked counter)
# ---------------------------------------------------------------------------


class TestChunkFile:
    def test_simple_headings(self):
        source = (
            b"---\ntitle: Guide\n---\n\n# Introduction\n\n"
            b"This is a guide about many things.\n\n"
            b"## Setup\n\nSetup instructions here with enough "
            b"words to be over the minimum token threshold for "
            b"proper chunking behavior and testing.\n\n"
            b"## Usage\n\nUsage instructions with sufficient "
            b"content to exceed minimum tokens so the section "
            b"stands alone as its own chunk."
        )
        chunks = _chunk_file_mocked(source, "guide.mdx")
        assert len(chunks) >= 1
        assert all(isinstance(c, DocChunk) for c in chunks)
        assert all(c.content_type == "section" for c in chunks)
        assert all(c.source_file == "guide.mdx" for c in chunks)

    def test_frontmatter_in_header_path(self):
        source = (
            b"---\ntitle: My Guide\n---\n\n# Introduction\n\n"
            b"Some content here that is long enough to not be "
            b"merged with anything else in the document.\n\n"
            b"## Details\n\nMore detailed content here that is "
            b"also sufficiently long to constitute its own chunk."
        )
        chunks = _chunk_file_mocked(source, "test.mdx")
        # At least one chunk should have "My Guide" in header_path
        paths = [c.header_path for c in chunks]
        assert any("My Guide" in p for p in paths)

    def test_changelog_detection(self):
        source = (
            b"---\ntitle: Changelog\n---\n\n"
            b'<Update label="v1.0" description="2026-01-01">'
            b"\nFirst release.\n</Update>"
        )
        chunks = _chunk_file_mocked(source, "changelog.mdx")
        assert len(chunks) == 1
        assert chunks[0].content_type == "changelog"
        assert "v1.0" in chunks[0].header_path

    def test_empty_file(self):
        chunks = _chunk_file_mocked(b"", "empty.md")
        assert chunks == []

    def test_no_headings(self):
        source = (
            b"Just some plain text without any headings. "
            b"It has multiple sentences for processing. "
            b"This should produce at least one chunk."
        )
        chunks = _chunk_file_mocked(source, "plain.md")
        assert len(chunks) >= 1
        assert chunks[0].content_type == "section"

    def test_content_hash_deterministic(self):
        source = b"# Title\n\nSome content."
        chunks1 = _chunk_file_mocked(source, "test.md")
        chunks2 = _chunk_file_mocked(source, "test.md")
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.content_hash == c2.content_hash

    def test_code_block_extraction(self):
        source = (
            b"# Docs\n\nSome intro text.\n\n"
            b"```python\nprint('hello')\n```\n\n"
            b"More text after the code block."
        )
        chunks = _chunk_file_mocked(source, "code.md")
        all_blocks = []
        for c in chunks:
            all_blocks.extend(c.code_blocks)
        assert any("print('hello')" in b for b in all_blocks)

    def test_overlap_not_in_content(self):
        # Create two sections that are large enough to stay separate
        section_a = "First section. " * 30 + "Last sentence of A."
        section_b = "Second section content. " * 30
        source = f"# Title\n\n## Section A\n\n{section_a}\n\n## Section B\n\n{section_b}".encode()
        chunks = _chunk_file_mocked(source, "overlap.md")
        for c in chunks:
            if c.overlap_prefix:
                # overlap_prefix should not be part of content
                assert c.content != c.overlap_prefix

    def test_mdx_components_processed(self):
        source = b"# Guide\n\n<Tip>Important tip here.</Tip>\n\nRegular content follows."
        chunks = _chunk_file_mocked(source, "mdx.mdx")
        all_content = " ".join(c.content for c in chunks)
        assert "<Tip>" not in all_content
        assert "Important tip here" in all_content

    def test_mdx_imports_removed(self):
        source = (
            b"import { VersionBadge } from "
            b"'/snippets/version-badge.mdx'\n\n"
            b"# Title\n\nContent here."
        )
        chunks = _chunk_file_mocked(source, "imports.mdx")
        all_content = " ".join(c.content for c in chunks)
        assert "VersionBadge" not in all_content


# ---------------------------------------------------------------------------
# Integration tests with real corpus files
# ---------------------------------------------------------------------------

CORPUS_SAMPLES = [
    "getting-started/quickstart.mdx",
    "servers/tools.mdx",
    "changelog.mdx",
    "CONTRIBUTING.md",
]


def _read_corpus_file(relpath: str) -> bytes:
    path = CORPUS_DOCS / relpath
    if not path.exists():
        pytest.skip(f"Corpus file not found: {path}")
    return path.read_bytes()


@pytest.mark.parametrize("relpath", CORPUS_SAMPLES)
def test_corpus_chunking_no_errors(relpath):
    """Smoke test: chunk real corpus files without crashing."""
    source = _read_corpus_file(relpath)
    with patch("src.indexing.markdown_chunker._count_tokens", side_effect=_mock_counter):
        chunks = chunk_file(source, relpath)
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, DocChunk)
        assert c.content.strip()
        assert c.content_hash
        assert c.source_file == relpath


@pytest.mark.parametrize("relpath", CORPUS_SAMPLES)
def test_corpus_chunks_have_header_path(relpath):
    """Every chunk should have a non-empty header_path."""
    source = _read_corpus_file(relpath)
    with patch("src.indexing.markdown_chunker._count_tokens", side_effect=_mock_counter):
        chunks = chunk_file(source, relpath)
    for c in chunks:
        assert c.header_path, f"Empty header_path in {relpath}: {c.content[:50]}"


def test_changelog_produces_changelog_type():
    """changelog.mdx should produce chunks with content_type='changelog'."""
    source = _read_corpus_file("changelog.mdx")
    with patch("src.indexing.markdown_chunker._count_tokens", side_effect=_mock_counter):
        chunks = chunk_file(source, "changelog.mdx")
    assert all(c.content_type == "changelog" for c in chunks)
    assert len(chunks) > 1  # Multiple versions


def test_tools_mdx_has_code_blocks():
    """servers/tools.mdx should produce chunks with code_blocks."""
    source = _read_corpus_file("servers/tools.mdx")
    with patch("src.indexing.markdown_chunker._count_tokens", side_effect=_mock_counter):
        chunks = chunk_file(source, "servers/tools.mdx")
    all_blocks = []
    for c in chunks:
        all_blocks.extend(c.code_blocks)
    assert len(all_blocks) > 0, "Expected code blocks in tools.mdx"
