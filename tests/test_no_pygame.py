"""Guard tests: the racing/ package must stay headless.

The core simulation is meant to run on servers, in CI, and inside training
loops with no display attached. If pygame (or matplotlib) ever sneaks into
racing/, importing the package would drag in SDL / GUI machinery, slow down
imports, and break headless environments. The renderers (watch.py, play.py,
viewer.py) live OUTSIDE the package as pure observers — that separation is a
hard architectural rule stated in racing/__init__.py, and these tests enforce
it mechanically.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import subprocess
import sys

import racing

# Resolve the racing/ package directory and the repo root from the imported
# package itself, so the tests work no matter where pytest is invoked from.
RACING_DIR = pathlib.Path(racing.__file__).resolve().parent
REPO_ROOT = RACING_DIR.parent

FORBIDDEN = ("pygame", "matplotlib")

ALL_MODULES = [
    "racing",
    "racing.config",
    "racing.track",
    "racing.car",
    "racing.sensors",
    "racing.brain",
    "racing.simulation",
    "racing.evolution",
    "racing.persistence",
]


def _racing_source_files() -> list[pathlib.Path]:
    """Every .py file under racing/, recursively (future subpackages too)."""
    files = sorted(RACING_DIR.rglob("*.py"))
    assert files, f"no .py files found under {RACING_DIR} — bad path resolution"
    return files


def test_source_text_has_no_forbidden_import_statements():
    """Static scan: the literal strings 'import pygame' / 'import matplotlib'
    must not appear in any racing/ source file.

    Why it matters: this is the cheapest tripwire — it catches the common
    forms (`import pygame`, `import pygame as pg`) before any code even runs,
    and fails with the exact offending file so the regression is obvious.
    """
    for path in _racing_source_files():
        text = path.read_text(encoding="utf-8")
        for lib in FORBIDDEN:
            assert f"import {lib}" not in text, (
                f"{path.relative_to(REPO_ROOT)} contains 'import {lib}' — "
                f"the racing package must stay headless"
            )


def test_ast_imports_never_reference_forbidden_libs():
    """AST scan: no Import/ImportFrom node in racing/ may target pygame or
    matplotlib, under ANY spelling.

    Why it matters: a raw substring check misses `from pygame import display`
    or `from matplotlib.pyplot import plot` (neither contains the substring
    'import pygame'/'import matplotlib'), and it can false-positive on
    docstrings. Parsing the AST checks the actual import statements the
    interpreter would execute — nothing more, nothing less.
    """
    for path in _racing_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # node.module is None for relative imports like `from . import x`
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top not in FORBIDDEN, (
                    f"{path.relative_to(REPO_ROOT)} line {node.lineno} imports "
                    f"'{name}' — the racing package must stay headless"
                )


def test_importing_racing_modules_does_not_load_pygame():
    """Runtime check: after importing every racing submodule, pygame and
    matplotlib must be absent from sys.modules.

    Why it matters: static scans can't see dynamic imports
    (importlib.import_module('pygame'), __import__, imports hidden inside
    module-level function calls). Watching sys.modules catches anything that
    actually executes at import time. We pop the libs first so a viewer test
    or user session that already loaded pygame can't mask a regression.
    """
    for lib in FORBIDDEN:
        sys.modules.pop(lib, None)

    for mod_name in ALL_MODULES:
        importlib.import_module(mod_name)

    for lib in FORBIDDEN:
        assert lib not in sys.modules, (
            f"importing the racing package loaded '{lib}' at runtime — "
            f"the core must stay headless"
        )


def test_fresh_interpreter_import_stays_headless():
    """Strongest check: a brand-new Python process imports every racing
    module and verifies pygame/matplotlib never enter sys.modules.

    Why it matters: in the pytest process the racing modules may already be
    cached in sys.modules, so re-importing them is a no-op and the in-process
    check above could pass vacuously. A fresh subprocess has an empty import
    cache, so every module-level statement in racing/ genuinely executes.
    """
    imports = "; ".join(f"import {m}" for m in ALL_MODULES)
    code = (
        "import sys; "
        f"{imports}; "
        "bad = [lib for lib in ('pygame', 'matplotlib') if lib in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),  # repo root on sys.path so `racing` is importable
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fresh-interpreter import loaded a GUI library or failed outright.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
