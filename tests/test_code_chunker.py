"""Tests for the AST-aware code chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.indexing.code_chunker import chunk_file

CORPUS_CODE = Path(__file__).parent.parent / "data" / "corpus" / "fastmcp" / "code"


# ---------------------------------------------------------------------------
# Unit tests with synthetic source
# ---------------------------------------------------------------------------


def test_simple_function():
    source = b'''\
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
    chunks = chunk_file(source, "example.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    f = funcs[0]
    assert f.symbol_name == "greet"
    assert f.signature == "def greet(name: str) -> str"
    assert f.docstring == "Say hello."
    assert f.start_line == 1
    assert f.end_line == 3
    assert f.module == "example"
    assert f.content_hash  # non-empty


def test_function_no_type_hints():
    source = b"""\
def add(a, b):
    return a + b
"""
    chunks = chunk_file(source, "math_utils.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].signature == "def add(a, b)"
    assert funcs[0].docstring is None


def test_function_args_kwargs():
    source = b"""\
def variadic(*args: int, **kwargs: str) -> None:
    pass
"""
    chunks = chunk_file(source, "util.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].signature is not None
    assert "*args" in funcs[0].signature
    assert "**kwargs" in funcs[0].signature


def test_decorated_function():
    source = b'''\
@staticmethod
def helper() -> None:
    """A static helper."""
    pass
'''
    chunks = chunk_file(source, "helpers.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].signature is not None
    assert "@staticmethod" in funcs[0].signature
    assert "@staticmethod" in funcs[0].source_code
    assert funcs[0].start_line == 1  # starts at decorator


def test_multiple_decorators():
    source = b"""\
@decorator_a
@decorator_b(option=True)
def multi_decorated():
    pass
"""
    chunks = chunk_file(source, "deco.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].signature is not None
    assert "@decorator_a" in funcs[0].signature
    assert "@decorator_b(option=True)" in funcs[0].signature
    assert funcs[0].start_line == 1


def test_class_basic():
    source = b'''\
class Dog:
    """A dog class."""

    def bark(self) -> str:
        """Make noise."""
        return "woof"

    def sit(self) -> None:
        pass
'''
    chunks = chunk_file(source, "animals.py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    methods = [c for c in chunks if c.chunk_type == "method"]

    assert len(classes) == 1
    assert classes[0].symbol_name == "Dog"
    assert classes[0].docstring == "A dog class."
    assert classes[0].signature == "class Dog"

    assert len(methods) == 2
    assert methods[0].symbol_name == "Dog.bark"
    assert methods[0].docstring == "Make noise."
    assert methods[1].symbol_name == "Dog.sit"


def test_class_with_inheritance():
    source = b"""\
from typing import Generic, TypeVar
T = TypeVar("T")

class Container(list, Generic[T]):
    pass
"""
    chunks = chunk_file(source, "containers.py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert len(classes) == 1
    assert classes[0].signature is not None
    assert "Container" in classes[0].signature
    assert "list, Generic[T]" in classes[0].signature


def test_class_with_property():
    source = b"""\
class Config:
    @property
    def value(self) -> int:
        return 42
"""
    chunks = chunk_file(source, "config.py")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert len(methods) == 1
    assert methods[0].symbol_name == "Config.value"
    assert methods[0].signature is not None
    assert "@property" in methods[0].signature


def test_nested_function_not_separate_chunk():
    source = b"""\
def outer():
    def inner():
        return 42
    return inner()
"""
    chunks = chunk_file(source, "nested.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert funcs[0].symbol_name == "outer"
    assert "inner" in funcs[0].source_code


def test_import_filtering():
    source = b"""\
import os
import sys
from pathlib import Path
from typing import Any, Optional

def read_file(p: Path) -> str:
    return p.read_text()
"""
    chunks = chunk_file(source, "io_utils.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    imports = funcs[0].imports
    # Should include Path but not os, sys, Any, Optional
    assert any("Path" in i for i in imports)
    assert not any(i.strip() == "import os" for i in imports)
    assert not any(i.strip() == "import sys" for i in imports)


def test_try_except_imports():
    source = b"""\
try:
    from fast_lib import FastThing
except ImportError:
    FastThing = None

def use_it() -> None:
    x = FastThing()
"""
    chunks = chunk_file(source, "compat.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert any("FastThing" in i for i in funcs[0].imports)


def test_type_checking_imports():
    source = b"""\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mylib import Secret

def process(data: str) -> None:
    x: Secret = data
"""
    chunks = chunk_file(source, "typed.py")
    funcs = [c for c in chunks if c.chunk_type == "function"]
    assert len(funcs) == 1
    assert any("Secret" in i for i in funcs[0].imports)


def test_module_chunk():
    source = b'''\
"""Module docstring."""

import os

def foo():
    pass
'''
    chunks = chunk_file(source, "simple.py")
    modules = [c for c in chunks if c.chunk_type == "module"]
    assert len(modules) == 1
    m = modules[0]
    assert m.module == "simple"
    assert m.symbol_name == "simple"
    assert m.docstring == "Module docstring."
    assert m.start_line == 1
    assert m.source_code == source.decode()
    assert "import os" in m.imports
    assert m.signature is None


def test_init_module_name():
    source = b"x = 1\n"
    chunks = chunk_file(source, "mypackage/__init__.py")
    modules = [c for c in chunks if c.chunk_type == "module"]
    assert modules[0].module == "mypackage"


def test_content_hash_deterministic():
    source = b"def f(): pass\n"
    chunks_a = chunk_file(source, "a.py")
    chunks_b = chunk_file(source, "b.py")
    # Same source_code -> same hash (for the function chunk)
    funcs_a = [c for c in chunks_a if c.chunk_type == "function"]
    funcs_b = [c for c in chunks_b if c.chunk_type == "function"]
    assert funcs_a[0].content_hash == funcs_b[0].content_hash


def test_decorated_class():
    source = b'''\
@dataclass(frozen=True)
class Point:
    """A 2D point."""
    x: int
    y: int
'''
    chunks = chunk_file(source, "geo.py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert len(classes) == 1
    assert classes[0].signature is not None
    assert "@dataclass(frozen=True)" in classes[0].signature
    assert classes[0].docstring == "A 2D point."
    assert classes[0].start_line == 1  # includes decorator


def test_empty_file():
    source = b""
    chunks = chunk_file(source, "empty.py")
    # Should produce at least a module chunk
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "module"


def test_dependencies_extraction():
    source = b"""\
from fastmcp.tools.tool import Tool, ToolResult

class MyTool(Tool):
    def run(self) -> ToolResult:
        return ToolResult()
"""
    chunks = chunk_file(source, "my_tool.py")
    classes = [c for c in chunks if c.chunk_type == "class"]
    assert len(classes) == 1
    assert "fastmcp.tools.tool.Tool" in classes[0].dependencies
    assert "fastmcp.tools.tool.ToolResult" in classes[0].dependencies


# ---------------------------------------------------------------------------
# Integration tests with real corpus files
# ---------------------------------------------------------------------------

CORPUS_SAMPLES = [
    "tools/function_tool.py",
    "tools/function_parsing.py",
    "__init__.py",
    "server/middleware/middleware.py",
]


@pytest.mark.parametrize("relpath", CORPUS_SAMPLES)
def test_corpus_file_chunking(relpath: str):
    """Smoke test: chunk real corpus files without errors."""
    filepath = CORPUS_CODE / relpath
    if not filepath.exists():
        pytest.skip(f"Corpus not extracted: {filepath}")

    source = filepath.read_bytes()
    chunks = chunk_file(source, f"fastmcp/{relpath}")

    # Basic invariants
    assert len(chunks) >= 1, "should produce at least a module chunk"

    module_chunks = [c for c in chunks if c.chunk_type == "module"]
    assert len(module_chunks) == 1, "exactly one module chunk per file"

    for c in chunks:
        assert c.chunk_type in ("function", "class", "method", "module")
        assert c.source_code, "source_code must be non-empty"
        assert c.content_hash, "content_hash must be non-empty"
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
        assert c.symbol_name, "symbol_name must be non-empty"


@pytest.mark.parametrize("relpath", CORPUS_SAMPLES)
def test_method_source_within_class(relpath: str):
    """Every method's source_code should appear within its parent class source."""
    filepath = CORPUS_CODE / relpath
    if not filepath.exists():
        pytest.skip(f"Corpus not extracted: {filepath}")

    source = filepath.read_bytes()
    chunks = chunk_file(source, f"fastmcp/{relpath}")

    classes = {c.symbol_name: c for c in chunks if c.chunk_type == "class"}
    methods = [c for c in chunks if c.chunk_type == "method"]

    for m in methods:
        # Method symbol is "ClassName.method_name"
        class_name = m.symbol_name.rsplit(".", 1)[0]
        if class_name in classes:
            assert m.source_code in classes[class_name].source_code, (
                f"Method {m.symbol_name} source not found in class {class_name}"
            )
