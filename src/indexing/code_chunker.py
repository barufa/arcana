"""AST-aware Python code chunker using tree-sitter.

Parses Python source files and extracts semantic code chunks at three
granularity levels: module, class, and function/method. Each chunk
includes rich metadata (signature, docstring, imports, dependencies)
for downstream embedding and retrieval.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from src.logger import log_operation, logger

PY_LANGUAGE = Language(tspython.language())


def _node_text(node: Node | None) -> str:
    """Safely decode a tree-sitter node's text, returning '' if None."""
    if node is None or node.text is None:
        return ""
    return node.text.decode()


@dataclass(frozen=True)
class CodeChunk:
    """A semantic code chunk extracted from a Python source file."""

    filepath: str
    module: str
    chunk_type: str  # 'function' | 'class' | 'method' | 'module'
    symbol_name: str
    signature: str | None
    docstring: str | None
    source_code: str
    imports: list[str]
    dependencies: list[str]
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    content_hash: str


@dataclass
class _ImportEntry:
    """An imported name and its full import line."""

    name: str
    line: str
    module_path: str | None  # for 'from X import Y', this is X


@dataclass
class _ImportIndex:
    """Index of all imports in a file, mapping names to import lines."""

    entries: dict[str, _ImportEntry] = field(default_factory=dict)
    all_lines: list[str] = field(default_factory=list)

    def add(self, name: str, line: str, module_path: str | None = None) -> None:
        if name not in self.entries:
            self.entries[name] = _ImportEntry(name=name, line=line, module_path=module_path)
        if line not in self.all_lines:
            self.all_lines.append(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filepath_to_module(filepath: str) -> str:
    """Convert a filepath to a dotted Python module name.

    Examples:
        'fastmcp/tools/function_tool.py' -> 'fastmcp.tools.function_tool'
        'fastmcp/tools/__init__.py' -> 'fastmcp.tools'
    """
    p = filepath.removesuffix(".py").replace("/", ".")
    if p.endswith(".__init__"):
        p = p[: -len(".__init__")]
    return p


def _content_hash(source_code: str) -> str:
    return hashlib.sha256(source_code.encode()).hexdigest()


def _extract_docstring(body_node: Node) -> str | None:
    """Extract docstring from the first statement in a body block."""
    if body_node.child_count == 0:
        return None
    first = body_node.children[0]
    if first.type != "expression_statement" or first.child_count == 0:
        return None
    string_node = first.children[0]
    if string_node.type != "string":
        return None
    return _strip_string_delimiters(_node_text(string_node))


def _build_function_signature(func_node: Node, decorators: list[Node] | None = None) -> str:
    """Build a signature string for a function/method definition."""
    parts: list[str] = []
    if decorators:
        for dec in decorators:
            parts.append(_node_text(dec))

    name_text = _node_text(func_node.child_by_field_name("name"))
    params_text = _node_text(func_node.child_by_field_name("parameters"))
    ret = func_node.child_by_field_name("return_type")

    sig = f"def {name_text}{params_text}"
    if ret:
        sig += f" -> {_node_text(ret)}"
    parts.append(sig)
    return "\n".join(parts)


def _build_class_signature(cls_node: Node, decorators: list[Node] | None = None) -> str:
    """Build a signature string for a class definition."""
    parts: list[str] = []
    if decorators:
        for dec in decorators:
            parts.append(_node_text(dec))

    name_text = _node_text(cls_node.child_by_field_name("name"))
    superclasses = cls_node.child_by_field_name("superclasses")

    sig = f"class {name_text}"
    if superclasses:
        sig += _node_text(superclasses)
    parts.append(sig)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Import collection
# ---------------------------------------------------------------------------


def _collect_imports(root_node: Node) -> _ImportIndex:
    """Walk the AST root to collect all imports, including conditional ones."""
    index = _ImportIndex()
    _collect_imports_from_children(root_node, index)
    return index


def _collect_imports_from_children(node: Node, index: _ImportIndex) -> None:
    """Recursively collect import statements from node's children."""
    for child in node.children:
        if child.type in ("import_statement", "import_from_statement", "future_import_statement"):
            _register_import(child, index)
        elif child.type == "if_statement":
            # Walk into `if TYPE_CHECKING:` blocks
            condition = child.child_by_field_name("condition")
            if condition and _node_text(condition).strip() == "TYPE_CHECKING":
                consequence = child.child_by_field_name("consequence")
                if consequence:
                    _collect_imports_from_children(consequence, index)
        elif child.type == "try_statement":
            # Walk into try/except blocks
            for sub in child.children:
                if sub.type == "block" or sub.type == "except_clause":
                    _collect_imports_from_children(sub, index)


def _register_import(node: Node, index: _ImportIndex) -> None:
    """Register an import statement's names into the index."""
    line = _node_text(node).strip()

    if node.type == "import_statement":
        # `import X` or `import X.Y.Z` or `import X as A`
        for child in node.children:
            if child.type == "dotted_name":
                top = _node_text(child).split(".")[0]
                index.add(top, line)
            elif child.type == "aliased_import":
                alias = child.child_by_field_name("alias")
                if alias:
                    index.add(_node_text(alias), line)
                else:
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        top = _node_text(name_node).split(".")[0]
                        index.add(top, line)

    elif node.type in ("import_from_statement", "future_import_statement"):
        # `from X.Y import a, b, c` or `from X.Y import *`
        module_path: str | None = None
        seen_import_keyword = False

        for child in node.children:
            if child.type == "from":
                continue
            elif child.type == "import":
                seen_import_keyword = True
                continue

            if not seen_import_keyword:
                if child.type == "dotted_name" and module_path is None:
                    module_path = _node_text(child)
                elif child.type == "relative_import":
                    module_path = _node_text(child)
            else:
                if child.type == "dotted_name":
                    index.add(_node_text(child), line, module_path)
                elif child.type == "aliased_import":
                    alias = child.child_by_field_name("alias")
                    if alias:
                        index.add(_node_text(alias), line, module_path)
                    else:
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            index.add(_node_text(name_node), line, module_path)
                elif child.type == "wildcard_import":
                    index.add("*", line, module_path)


# ---------------------------------------------------------------------------
# Import filtering and dependency extraction
# ---------------------------------------------------------------------------

# Cache compiled regexes for import name matching
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _name_used_in(name: str, source_code: str) -> bool:
    """Check if an imported name appears in source code as a whole word."""
    if name == "*":
        return True  # wildcard imports are always included
    pat = _WORD_BOUNDARY_CACHE.get(name)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(name)}\b")
        _WORD_BOUNDARY_CACHE[name] = pat
    return pat.search(source_code) is not None


def _filter_imports(source_code: str, imports_index: _ImportIndex) -> list[str]:
    """Return the import lines that the chunk actually uses."""
    used_lines: set[str] = set()
    for entry in imports_index.entries.values():
        if _name_used_in(entry.name, source_code):
            used_lines.add(entry.line)
    # Preserve original order
    return [line for line in imports_index.all_lines if line in used_lines]


def _extract_dependencies(source_code: str, imports_index: _ImportIndex) -> list[str]:
    """Extract fully qualified internal references from the chunk's source."""
    deps: set[str] = set()
    for entry in imports_index.entries.values():
        if entry.module_path and _name_used_in(entry.name, source_code):
            deps.add(f"{entry.module_path}.{entry.name}")
    return sorted(deps)


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------


def _get_source_text(source: bytes, start_byte: int, end_byte: int) -> str:
    return source[start_byte:end_byte].decode()


def _extract_function_chunk(
    func_node: Node,
    source: bytes,
    filepath: str,
    module: str,
    imports_index: _ImportIndex,
    chunk_type: str,
    decorators: list[Node] | None = None,
    class_name: str | None = None,
) -> CodeChunk:
    """Extract a function or method chunk."""
    # Source range includes decorators
    start_byte = decorators[0].start_byte if decorators else func_node.start_byte
    end_byte = func_node.end_byte
    source_code = _get_source_text(source, start_byte, end_byte)

    symbol_name = _node_text(func_node.child_by_field_name("name"))

    body = func_node.child_by_field_name("body")
    docstring = _extract_docstring(body) if body else None

    start_line = (decorators[0].start_point[0] if decorators else func_node.start_point[0]) + 1
    end_line = func_node.end_point[0] + 1

    # For methods, prefix symbol_name with class name for display in signatures
    display_name = f"{class_name}.{symbol_name}" if class_name else symbol_name

    return CodeChunk(
        filepath=filepath,
        module=module,
        chunk_type=chunk_type,
        symbol_name=display_name,
        signature=_build_function_signature(func_node, decorators),
        docstring=docstring,
        source_code=source_code,
        imports=_filter_imports(source_code, imports_index),
        dependencies=_extract_dependencies(source_code, imports_index),
        start_line=start_line,
        end_line=end_line,
        content_hash=_content_hash(source_code),
    )


def _extract_class_chunk(
    cls_node: Node,
    source: bytes,
    filepath: str,
    module: str,
    imports_index: _ImportIndex,
    decorators: list[Node] | None = None,
) -> list[CodeChunk]:
    """Extract a class chunk plus method chunks for each method."""
    chunks: list[CodeChunk] = []

    # Class chunk (full class including decorators)
    start_byte = decorators[0].start_byte if decorators else cls_node.start_byte
    end_byte = cls_node.end_byte
    source_code = _get_source_text(source, start_byte, end_byte)

    class_name = _node_text(cls_node.child_by_field_name("name"))

    body = cls_node.child_by_field_name("body")
    docstring = _extract_docstring(body) if body else None

    start_line = (decorators[0].start_point[0] if decorators else cls_node.start_point[0]) + 1
    end_line = cls_node.end_point[0] + 1

    chunks.append(
        CodeChunk(
            filepath=filepath,
            module=module,
            chunk_type="class",
            symbol_name=class_name,
            signature=_build_class_signature(cls_node, decorators),
            docstring=docstring,
            source_code=source_code,
            imports=_filter_imports(source_code, imports_index),
            dependencies=_extract_dependencies(source_code, imports_index),
            start_line=start_line,
            end_line=end_line,
            content_hash=_content_hash(source_code),
        )
    )

    # Extract methods from class body
    if body:
        for child in body.children:
            if child.type == "function_definition":
                chunks.append(
                    _extract_function_chunk(
                        child,
                        source,
                        filepath,
                        module,
                        imports_index,
                        chunk_type="method",
                        class_name=class_name,
                    )
                )
            elif child.type == "decorated_definition":
                defn = child.child_by_field_name("definition")
                if defn and defn.type == "function_definition":
                    method_decorators = [c for c in child.children if c.type == "decorator"]
                    chunks.append(
                        _extract_function_chunk(
                            defn,
                            source,
                            filepath,
                            module,
                            imports_index,
                            chunk_type="method",
                            decorators=method_decorators,
                            class_name=class_name,
                        )
                    )

    return chunks


def _extract_module_chunk(
    source: bytes,
    filepath: str,
    module: str,
    imports_index: _ImportIndex,
    root_node: Node,
) -> CodeChunk:
    """Extract the whole-file module chunk."""
    source_code = source.decode()

    # Module docstring
    docstring = None
    if root_node.child_count > 0:
        first = root_node.children[0]
        if first.type == "expression_statement" and first.child_count > 0:
            string_node = first.children[0]
            if string_node.type == "string":
                docstring = _strip_string_delimiters(_node_text(string_node))

    total_lines = source_code.count("\n") + 1 if source_code else 1

    return CodeChunk(
        filepath=filepath,
        module=module,
        chunk_type="module",
        symbol_name=module,
        signature=None,
        docstring=docstring,
        source_code=source_code,
        imports=list(imports_index.all_lines),
        dependencies=_extract_dependencies(source_code, imports_index),
        start_line=1,
        end_line=total_lines,
        content_hash=_content_hash(source_code),
    )


def _strip_string_delimiters(raw: str) -> str | None:
    """Strip quote delimiters from a Python string literal."""
    for delim in ('"""', "'''"):
        if raw.startswith(delim) and raw.endswith(delim):
            return raw[3:-3].strip()
    for delim in ('"', "'"):
        if raw.startswith(delim) and raw.endswith(delim):
            return raw[1:-1].strip()
    return raw or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(source: bytes, filepath: str) -> list[CodeChunk]:
    """Parse a Python file and extract semantic code chunks.

    Produces chunks at three granularity levels:
    - **module**: the entire file
    - **class**: each class with all its methods
    - **method**: each individual method within a class
    - **function**: each top-level function

    Args:
        source: Raw bytes of the Python source file.
        filepath: Relative filepath within the corpus
            (e.g. "fastmcp/tools/function_tool.py").

    Returns:
        List of CodeChunk instances.
    """
    module = _filepath_to_module(filepath)

    with log_operation("chunk_file", metadata={"filepath": filepath}):
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(source)

        if tree.root_node.has_error:
            logger.warning(f"Parse errors in {filepath}, extracting partial AST")

        # Pass 1: collect imports
        imports_index = _collect_imports(tree.root_node)

        # Pass 2: extract definition chunks
        chunks: list[CodeChunk] = []
        counts = {"function": 0, "class": 0, "method": 0}

        for child in tree.root_node.children:
            if child.type == "function_definition":
                chunks.append(
                    _extract_function_chunk(
                        child, source, filepath, module, imports_index, chunk_type="function"
                    )
                )
                counts["function"] += 1

            elif child.type == "class_definition":
                class_chunks = _extract_class_chunk(child, source, filepath, module, imports_index)
                chunks.extend(class_chunks)
                counts["class"] += 1
                counts["method"] += sum(1 for c in class_chunks if c.chunk_type == "method")

            elif child.type == "decorated_definition":
                defn = child.child_by_field_name("definition")
                if defn is None:
                    continue
                decs = [c for c in child.children if c.type == "decorator"]

                if defn.type == "function_definition":
                    chunks.append(
                        _extract_function_chunk(
                            defn,
                            source,
                            filepath,
                            module,
                            imports_index,
                            chunk_type="function",
                            decorators=decs,
                        )
                    )
                    counts["function"] += 1

                elif defn.type == "class_definition":
                    class_chunks = _extract_class_chunk(
                        defn, source, filepath, module, imports_index, decorators=decs
                    )
                    chunks.extend(class_chunks)
                    counts["class"] += 1
                    counts["method"] += sum(1 for c in class_chunks if c.chunk_type == "method")

        # Module chunk (always produced)
        module_chunk = _extract_module_chunk(
            source, filepath, module, imports_index, tree.root_node
        )
        chunks.insert(0, module_chunk)

        logger.bind(
            filepath=filepath,
            chunk_counts=counts,
            total_chunks=len(chunks),
        ).info("file chunked")

    return chunks
