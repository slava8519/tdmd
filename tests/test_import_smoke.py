from __future__ import annotations

import importlib
import py_compile
from pathlib import Path


def test_import_smoke():
    importlib.import_module("tdmd.td_automaton")
    importlib.import_module("tdmd.td_full_mpi")
    importlib.import_module("tdmd.main")


def test_py_compile_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "tdmd" / "td_automaton.py",
        root / "tdmd" / "td_full_mpi.py",
        root / "tdmd" / "main.py",
    ]
    for src in targets:
        py_compile.compile(str(src), cfile=str(tmp_path / f"{src.name}c"), doraise=True)
