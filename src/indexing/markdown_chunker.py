"""Semantic markdown/MDX documentation chunker.

Parses markdown files using mistune's AST mode, builds a heading-based
section hierarchy, and produces doc chunks with recursive merge/split
logic based on token counts from Voyage AI's tokenizer.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mistune
import yaml

from src.logger import log_operation, logger

# ---------------------------------------------------------------------------
# Voyage AI token counter (lazy singleton)
# ---------------------------------------------------------------------------

_voyage_client: Any = None


def _get_voyage_client() -> Any:
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        _voyage_client = voyageai.Client()
    return _voyage_client


def _count_tokens(text: str) -> int:
    """Count tokens using Voyage AI's tokenizer."""
    if not text:
        return 0
    return _get_voyage_client().count_tokens([text])


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocChunk:
    """A semantic documentation chunk extracted from a markdown file."""

    source_file: str  # relative path (e.g. "getting-started/quickstart.mdx")
    content_type: str  # 'section' | 'changelog'
    header_path: str  # "Title > Section > Subsection"
    scope: str | None  # reserved for proposition_extractor
    content: str  # text for the LLM
    original_section: str  # full original markdown of the section
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    content_hash: str  # SHA-256 of original_section
    overlap_prefix: str | None  # last 2-3 sentences of previous section (embedding only)
    code_blocks: list[str]  # extracted fenced code blocks for dual indexing


# ---------------------------------------------------------------------------
# Internal section tree
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    """Mutable node in the heading hierarchy (internal use only)."""

    level: int  # 0=root, 1-6=heading levels
    title: str
    tokens: list[dict] = field(default_factory=list)
    children: list[_Section] = field(default_factory=list)
    start_line: int = 1
    end_line: int = 1
    parent: _Section | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    """SHA-256 hash of text for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract YAML frontmatter and return (metadata, remaining text)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    try:
        metadata = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        metadata = None
    return metadata, text[m.end() :]


_MDX_IMPORT_RE = re.compile(r"^import\s+\{[^}]*\}\s+from\s+['\"].*?['\"]\s*;?\s*$", re.MULTILINE)


def _strip_mdx_imports(text: str) -> str:
    """Remove MDX import statements."""
    return _MDX_IMPORT_RE.sub("", text)


# MDX component patterns — order matters: inner before outer
_MDX_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    # Self-closing tags
    (re.compile(r"<VersionBadge\s+version=[\"']([^\"']+)[\"']\s*/?>", re.DOTALL), r"[Since v\1]"),
    # Content-wrapping tags → markdown equivalents
    (re.compile(r"<Tip>\s*(.*?)\s*</Tip>", re.DOTALL), r"> **Tip:** \1"),
    (re.compile(r"<Warning>\s*(.*?)\s*</Warning>", re.DOTALL), r"> **Warning:** \1"),
    (re.compile(r"<Note>\s*(.*?)\s*</Note>", re.DOTALL), r"> **Note:** \1"),
    (re.compile(r"<Info>\s*(.*?)\s*</Info>", re.DOTALL), r"> **Info:** \1"),
    # Card → heading + content
    (
        re.compile(r'<Card[^>]*title=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</Card>', re.DOTALL),
        r"#### \1\n\n\2",
    ),
    # ParamField → list item
    (
        re.compile(
            r'<ParamField[^>]*name=["\']([^"\']+)["\'][^>]*type=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</ParamField>',
            re.DOTALL,
        ),
        r"- **\1** (*\2*): \3",
    ),
    (
        re.compile(
            r'<ParamField[^>]*name=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</ParamField>',
            re.DOTALL,
        ),
        r"- **\1**: \2",
    ),
    # ResponseField → list item
    (
        re.compile(
            r'<ResponseField[^>]*name=["\']([^"\']+)["\'][^>]*type=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</ResponseField>',
            re.DOTALL,
        ),
        r"- **\1** (*\2*): \3",
    ),
    # Wrappers that just pass through content
    (re.compile(r"<CodeGroup>\s*(.*?)\s*</CodeGroup>", re.DOTALL), r"\1"),
    (re.compile(r"<Steps>\s*(.*?)\s*</Steps>", re.DOTALL), r"\1"),
    (re.compile(r"<Accordion[^>]*>\s*(.*?)\s*</Accordion>", re.DOTALL), r"\1"),
    # Catch-all: strip unknown self-closing MDX tags
    (re.compile(r"<[A-Z][A-Za-z]*\s+[^>]*/\s*>"), ""),
    # Catch-all: unwrap unknown block-level MDX tags (keep content)
    (re.compile(r"<([A-Z][A-Za-z]*)[^>]*>\s*(.*?)\s*</\1>", re.DOTALL), r"\2"),
]


def _strip_mdx_components(text: str) -> str:
    """Convert MDX components to standard markdown equivalents."""
    for pattern, replacement in _MDX_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Changelog parser
# ---------------------------------------------------------------------------

_UPDATE_RE = re.compile(
    r'<Update\s+label=["\']([^"\']+)["\']\s+description=["\']([^"\']+)["\'][^>]*>\s*(.*?)\s*</Update>',
    re.DOTALL,
)


def _parse_changelog_updates(text: str) -> list[tuple[str, str, str]]:
    """Extract <Update> blocks → list of (label, description, inner_content)."""
    return [(m.group(1), m.group(2), m.group(3).strip()) for m in _UPDATE_RE.finditer(text)]


def _chunk_changelog(
    text: str,
    filepath: str,
    frontmatter: dict[str, Any] | None,
    source_lines: list[str],
) -> list[DocChunk]:
    """Generate one DocChunk per changelog <Update> entry."""
    doc_title = (frontmatter or {}).get("title", "Changelog")
    updates = _parse_changelog_updates(text)
    chunks: list[DocChunk] = []

    for label, _description, inner in updates:
        # Clean MDX inside the update content
        clean_inner = _strip_mdx_imports(inner)
        clean_inner = _strip_mdx_components(clean_inner)

        # Find approximate line numbers by searching raw text
        start_line = 1
        end_line = len(source_lines)
        marker = f'label="{label}"'
        alt_marker = f"label='{label}'"
        for i, line in enumerate(source_lines, 1):
            if marker in line or alt_marker in line:
                start_line = i
                break
        closing = "</Update>"
        for i, line in enumerate(source_lines[start_line - 1 :], start_line):
            if closing in line:
                end_line = i
                break

        chunks.append(
            DocChunk(
                source_file=filepath,
                content_type="changelog",
                header_path=f"{doc_title} > {label}",
                scope=None,
                content=clean_inner,
                original_section=inner,
                start_line=start_line,
                end_line=end_line,
                content_hash=_content_hash(inner),
                overlap_prefix=None,
                code_blocks=_extract_code_blocks_from_text(clean_inner),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_code_blocks_from_text(text: str) -> list[str]:
    """Extract fenced code block contents from markdown text."""
    return [m.group(1).strip() for m in _FENCED_CODE_RE.finditer(text) if m.group(1).strip()]


def _extract_code_blocks_from_tokens(tokens: list[dict]) -> list[str]:
    """Extract code blocks from mistune AST tokens."""
    blocks: list[str] = []
    for tok in tokens:
        if tok.get("type") == "block_code":
            raw = tok.get("raw", "").strip()
            if raw:
                blocks.append(raw)
        if "children" in tok:
            blocks.extend(_extract_code_blocks_from_tokens(tok["children"]))
    return blocks


# ---------------------------------------------------------------------------
# Table normalization
# ---------------------------------------------------------------------------


def _table_to_text(token: dict) -> str:
    """Convert a mistune table AST token to structured text.

    Output format:
        Header1 | Header2 | Header3
        Row 1: Header1=Val1, Header2=Val2, Header3=Val3
    """
    children = token.get("children", [])
    if not children:
        return ""

    headers: list[str] = []
    rows: list[list[str]] = []

    for child in children:
        child_type = child.get("type", "")
        if child_type == "table_head":
            for cell in child.get("children", []):
                headers.append(_extract_inline_text(cell.get("children", [])))
        elif child_type == "table_body":
            for row in child.get("children", []):
                cells = []
                for cell in row.get("children", []):
                    cells.append(_extract_inline_text(cell.get("children", [])))
                rows.append(cells)

    lines = []
    if headers:
        lines.append(" | ".join(headers))
    for i, row in enumerate(rows, 1):
        pairs = []
        for j, cell in enumerate(row):
            hdr = headers[j] if j < len(headers) else f"Col{j + 1}"
            pairs.append(f"{hdr}={cell}")
        lines.append(f"Row {i}: {', '.join(pairs)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text extraction from tokens
# ---------------------------------------------------------------------------


def _extract_inline_text(tokens: list[dict]) -> str:
    """Recursively extract plain text from inline tokens."""
    parts: list[str] = []
    for tok in tokens:
        if "raw" in tok:
            parts.append(tok["raw"])
        elif "children" in tok:
            parts.append(_extract_inline_text(tok["children"]))
        elif "text" in tok:
            parts.append(tok["text"])
    return "".join(parts)


def _extract_text(tokens: list[dict]) -> str:
    """Extract plain text from a list of block-level AST tokens."""
    parts: list[str] = []
    for tok in tokens:
        t = tok.get("type", "")
        if t == "paragraph":
            children = tok.get("children", [])
            if children:
                parts.append(_extract_inline_text(children))
            elif "text" in tok:
                parts.append(tok["text"])
        elif t == "heading":
            children = tok.get("children", [])
            if children:
                parts.append(_extract_inline_text(children))
            elif "text" in tok:
                parts.append(tok["text"])
        elif t == "block_code":
            parts.append(tok.get("raw", ""))
        elif t == "table":
            parts.append(_table_to_text(tok))
        elif t in ("list", "block_quote"):
            children = tok.get("children", [])
            if children:
                parts.append(_extract_text(children))
        elif t == "list_item":
            children = tok.get("children", [])
            if children:
                parts.append(_extract_text(children))
        elif "children" in tok:
            parts.append(_extract_text(tok["children"]))
        elif "text" in tok:
            parts.append(tok["text"])
        elif "raw" in tok:
            parts.append(tok["raw"])
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Markdown reconstruction from tokens
# ---------------------------------------------------------------------------


def _tokens_to_markdown(tokens: list[dict]) -> str:
    """Reconstruct readable markdown from mistune AST tokens."""
    parts: list[str] = []
    for tok in tokens:
        t = tok.get("type", "")
        if t == "paragraph":
            children = tok.get("children", [])
            if children:
                parts.append(_inline_to_markdown(children))
            elif "text" in tok:
                parts.append(tok["text"])
        elif t == "heading":
            level = tok.get("attrs", {}).get("level", 1)
            children = tok.get("children", [])
            text = _inline_to_markdown(children) if children else tok.get("text", "")
            parts.append(f"{'#' * level} {text}")
        elif t == "block_code":
            info = tok.get("attrs", {}).get("info", "") or ""
            raw = tok.get("raw", "")
            parts.append(f"```{info}\n{raw}```")
        elif t == "block_quote":
            children = tok.get("children", [])
            inner = _tokens_to_markdown(children) if children else ""
            quoted = "\n".join(f"> {line}" for line in inner.split("\n"))
            parts.append(quoted)
        elif t == "list":
            children = tok.get("children", [])
            ordered = tok.get("attrs", {}).get("ordered", False)
            for idx, item in enumerate(children, 1):
                item_children = item.get("children", [])
                item_text = _tokens_to_markdown(item_children)
                prefix = f"{idx}." if ordered else "-"
                parts.append(f"{prefix} {item_text}")
        elif t == "table":
            parts.append(_table_to_text(tok))
        elif t == "thematic_break":
            parts.append("---")
        elif t in ("list_item",):
            children = tok.get("children", [])
            parts.append(_tokens_to_markdown(children))
        elif "children" in tok:
            parts.append(_tokens_to_markdown(tok["children"]))
        elif "raw" in tok:
            parts.append(tok["raw"])
        elif "text" in tok:
            parts.append(tok["text"])
    return "\n\n".join(p for p in parts if p)


def _inline_to_markdown(tokens: list[dict]) -> str:
    """Reconstruct inline markdown from inline tokens."""
    parts: list[str] = []
    for tok in tokens:
        t = tok.get("type", "")
        if t == "text":
            parts.append(tok.get("raw", tok.get("children", tok.get("text", ""))))
            if isinstance(parts[-1], list):
                parts[-1] = _inline_to_markdown(parts[-1])
        elif t == "codespan":
            parts.append(f"`{tok.get('raw', tok.get('text', ''))}`")
        elif t == "strong":
            children = tok.get("children", [])
            inner = _inline_to_markdown(children) if children else tok.get("text", "")
            parts.append(f"**{inner}**")
        elif t == "emphasis":
            children = tok.get("children", [])
            inner = _inline_to_markdown(children) if children else tok.get("text", "")
            parts.append(f"*{inner}*")
        elif t == "link":
            children = tok.get("children", [])
            inner = _inline_to_markdown(children) if children else tok.get("text", "")
            href = tok.get("attrs", {}).get("url", tok.get("link", ""))
            parts.append(f"[{inner}]({href})")
        elif t == "image":
            alt = tok.get("attrs", {}).get("alt", "")
            src = tok.get("attrs", {}).get("url", tok.get("src", ""))
            parts.append(f"![{alt}]({src})")
        elif t == "softbreak":
            parts.append("\n")
        elif t == "linebreak":
            parts.append("\n")
        elif "children" in tok:
            parts.append(_inline_to_markdown(tok["children"]))
        elif "raw" in tok:
            parts.append(tok["raw"])
        elif "text" in tok:
            parts.append(tok["text"])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Section tree builder
# ---------------------------------------------------------------------------


def _build_section_tree(tokens: list[dict], source_lines: list[str]) -> _Section:
    """Build a heading hierarchy from mistune's flat token list.

    Uses a stack-based algorithm: each heading pops the stack until
    finding a parent with a lower level, then pushes itself.
    """
    root = _Section(level=0, title="(root)", start_line=1, end_line=len(source_lines))
    stack: list[_Section] = [root]
    line_cursor = 1

    for tok in tokens:
        if tok.get("type") == "heading":
            level = tok.get("attrs", {}).get("level", 1)
            children = tok.get("children", [])
            title = _extract_inline_text(children) if children else tok.get("text", "")

            # Find the heading line in source
            heading_line = _find_heading_line(source_lines, title, level, line_cursor)
            if heading_line > 0:
                line_cursor = heading_line

            new_section = _Section(
                level=level,
                title=title,
                start_line=heading_line if heading_line > 0 else line_cursor,
                end_line=len(source_lines),
            )

            # Pop until we find a proper parent
            while len(stack) > 1 and stack[-1].level >= level:
                popped = stack.pop()
                # Set end_line of popped section
                popped.end_line = max(popped.start_line, new_section.start_line - 1)

            stack[-1].children.append(new_section)
            new_section.parent = stack[-1]
            stack.append(new_section)
        else:
            # Non-heading token belongs to current section
            stack[-1].tokens.append(tok)

    # Close remaining sections
    for section in stack[1:]:
        section.end_line = len(source_lines)

    return root


def _find_heading_line(source_lines: list[str], title: str, level: int, start_from: int) -> int:
    """Find the line number of a heading in the source."""
    prefix = "#" * level + " "
    # Try exact match first
    for i in range(start_from - 1, len(source_lines)):
        line = source_lines[i].strip()
        if line.startswith(prefix) and title in line:
            return i + 1  # 1-indexed
    # Fallback: partial match
    title_lower = title.lower()
    for i in range(start_from - 1, len(source_lines)):
        line = source_lines[i].strip().lower()
        if line.startswith(prefix.lower()) and title_lower in line:
            return i + 1
    return 0


def _build_header_path(section: _Section, doc_title: str | None) -> str:
    """Build breadcrumb path from section up to root."""
    parts: list[str] = []
    current: _Section | None = section
    while current is not None:
        if current.level > 0:
            parts.append(current.title)
        elif doc_title:
            parts.append(doc_title)
        current = current.parent
    parts.reverse()
    return " > ".join(parts) if parts else "(untitled)"


# ---------------------------------------------------------------------------
# Merge / split chunking
# ---------------------------------------------------------------------------


def _collect_section_text(section: _Section) -> str:
    """Get the markdown text of a section's own tokens (excluding children)."""
    return _tokens_to_markdown(section.tokens)


def _collect_all_tokens(section: _Section) -> list[dict]:
    """Collect all tokens from a section and its children recursively."""
    all_tokens = list(section.tokens)
    for child in section.children:
        # Re-add the heading token for the child
        all_tokens.append(
            {
                "type": "heading",
                "children": [{"type": "text", "raw": child.title}],
                "attrs": {"level": child.level},
            }
        )
        all_tokens.extend(_collect_all_tokens(child))
    return all_tokens


def _chunk_section(
    section: _Section,
    *,
    min_tokens: int,
    max_tokens: int,
    counter: Callable[[str], int],
    doc_title: str | None,
) -> list[tuple[_Section, str, str]]:
    """Recursively chunk a section tree.

    Returns list of (section, content_text, original_text) tuples.
    """
    # First, recursively process children
    child_results: list[tuple[_Section, str, str]] = []
    for child in section.children:
        child_results.extend(
            _chunk_section(
                child,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                counter=counter,
                doc_title=doc_title,
            )
        )

    own_text = _collect_section_text(section)
    own_count = counter(own_text)

    # Leaf section (no children)
    if not section.children:
        if own_count < min_tokens and section.parent and section.parent.level > 0:
            # Too small — will be merged by parent; return as-is for parent to handle
            return [(section, own_text, own_text)]
        if own_count > max_tokens:
            return _split_large_section(
                section, own_text, max_tokens=max_tokens, counter=counter, doc_title=doc_title
            )
        return [(section, own_text, own_text)]

    # Section with children: merge small children into parent
    merged_tokens = list(section.tokens)
    standalone: list[tuple[_Section, str, str]] = []

    for child_section, child_text, child_original in child_results:
        child_count = counter(child_text)
        if child_count < min_tokens:
            # Merge small child into this section
            merged_tokens.append(
                {
                    "type": "heading",
                    "children": [{"type": "text", "raw": child_section.title}],
                    "attrs": {"level": child_section.level},
                }
            )
            merged_tokens.extend(child_section.tokens)
        else:
            standalone.append((child_section, child_text, child_original))

    merged_text = _tokens_to_markdown(merged_tokens)
    merged_count = counter(merged_text)

    result: list[tuple[_Section, str, str]] = []

    if merged_count > 0:
        if merged_count > max_tokens:
            merged_section = _Section(
                level=section.level,
                title=section.title,
                tokens=merged_tokens,
                start_line=section.start_line,
                end_line=section.end_line,
                parent=section.parent,
            )
            result.extend(
                _split_large_section(
                    merged_section,
                    merged_text,
                    max_tokens=max_tokens,
                    counter=counter,
                    doc_title=doc_title,
                )
            )
        else:
            result.append((section, merged_text, merged_text))

    result.extend(standalone)
    return result


def _split_large_section(
    section: _Section,
    text: str,
    *,
    max_tokens: int,
    counter: Callable[[str], int],
    doc_title: str | None,
) -> list[tuple[_Section, str, str]]:
    """Split a large section at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) <= 1:
        # Can't split further, return as-is
        return [(section, text, text)]

    results: list[tuple[_Section, str, str]] = []
    current_parts: list[str] = []
    current_count = 0
    part_num = 1

    for para in paragraphs:
        para_count = counter(para)
        if current_count + para_count > max_tokens and current_parts:
            # Emit current accumulation
            chunk_text = "\n\n".join(current_parts)
            part_section = _Section(
                level=section.level,
                title=(
                    f"{section.title} (part {part_num})"
                    if part_num > 1 or len(paragraphs) > 1
                    else section.title
                ),
                tokens=[],
                start_line=section.start_line,
                end_line=section.end_line,
                parent=section.parent,
            )
            results.append((part_section, chunk_text, chunk_text))
            current_parts = [para]
            current_count = para_count
            part_num += 1
        else:
            current_parts.append(para)
            current_count += para_count

    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        part_section = _Section(
            level=section.level,
            title=f"{section.title} (part {part_num})" if part_num > 1 else section.title,
            tokens=[],
            start_line=section.start_line,
            end_line=section.end_line,
            parent=section.parent,
        )
        results.append((part_section, chunk_text, chunk_text))

    return results


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def _extract_last_sentences(text: str, count: int = 3) -> str | None:
    """Extract the last N sentences from text."""
    text = text.strip()
    if not text:
        return None
    sentences = _SENTENCE_RE.split(text)
    if len(sentences) <= 1:
        return None
    last = sentences[-count:] if len(sentences) >= count else sentences
    return " ".join(s.strip() for s in last if s.strip()) or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(
    source: bytes,
    filepath: str,
    *,
    min_tokens: int = 100,
    max_tokens: int = 800,
) -> list[DocChunk]:
    """Parse a markdown/MDX file and produce semantic documentation chunks.

    Args:
        source: Raw file bytes.
        filepath: Relative path within the corpus.
        min_tokens: Sections below this merge with parent (default 100).
        max_tokens: Sections above this get split (default 800).

    Returns:
        List of DocChunk objects in document order.
    """
    with log_operation("chunk_markdown", metadata={"filepath": filepath}):
        text = source.decode("utf-8", errors="replace")
        source_lines = text.split("\n")

        # Step 1: Extract frontmatter
        frontmatter, clean_text = _strip_frontmatter(text)
        doc_title = (frontmatter or {}).get("title")

        # Step 2: Detect changelog
        if "<Update label=" in text:
            logger.debug(f"Detected changelog: {filepath}")
            return _chunk_changelog(text, filepath, frontmatter, source_lines)

        # Step 3: Strip MDX
        clean_text = _strip_mdx_imports(clean_text)
        clean_text = _strip_mdx_components(clean_text)

        # Step 4: Parse with mistune
        md = mistune.create_markdown(renderer="ast", plugins=["table"])
        tokens = md(clean_text)
        if not tokens:
            return []

        # Step 5: Build section tree
        # Recompute source_lines from clean_text for accurate line tracking
        clean_lines = clean_text.split("\n")
        tree = _build_section_tree(tokens, clean_lines)

        # Step 6: Recursive merge/split
        counter = _count_tokens
        raw_chunks = _chunk_section(
            tree,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            counter=counter,
            doc_title=doc_title,
        )

        # Step 7: Convert to DocChunk objects
        chunks: list[DocChunk] = []
        for section, content, original in raw_chunks:
            if not content.strip():
                continue
            header_path = _build_header_path(section, doc_title)
            code_blocks = _extract_code_blocks_from_text(content)
            chunks.append(
                DocChunk(
                    source_file=filepath,
                    content_type="section",
                    header_path=header_path,
                    scope=None,
                    content=content,
                    original_section=original,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    content_hash=_content_hash(original),
                    overlap_prefix=None,
                    code_blocks=code_blocks,
                )
            )

        # Step 8: Apply overlap prefixes
        for i in range(1, len(chunks)):
            overlap = _extract_last_sentences(chunks[i - 1].content)
            if overlap:
                # Replace with new instance (frozen dataclass)
                old = chunks[i]
                chunks[i] = DocChunk(
                    source_file=old.source_file,
                    content_type=old.content_type,
                    header_path=old.header_path,
                    scope=old.scope,
                    content=old.content,
                    original_section=old.original_section,
                    start_line=old.start_line,
                    end_line=old.end_line,
                    content_hash=old.content_hash,
                    overlap_prefix=overlap,
                    code_blocks=old.code_blocks,
                )

        logger.debug(f"Produced {len(chunks)} chunks from {filepath}")
        return chunks
