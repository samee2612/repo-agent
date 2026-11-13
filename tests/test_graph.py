"""Tests for AST-based NetworkX dependency graph."""

from pathlib import Path
from app.graph import build_dependency_graph


def test_graph_records_imports_calls_and_unresolved(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "def target():\n    return len('abc')\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from a import target\n\ndef caller():\n    return target()\n",
        encoding="utf-8",
    )

    graph = build_dependency_graph(tmp_path)

    assert graph.has_edge("b.py", "a.py::target")
    assert graph.edges["b.py", "a.py::target"]["relation"] == "imports"

    assert graph.has_edge("b.py::caller", "a.py::target")
    assert graph.edges["b.py::caller", "a.py::target"]["relation"] == "calls"

    # `len` is a built-in (unresolved against repo symbols), so no node/edge is created.
    assert "len" not in graph
    unresolved = graph.graph["unresolved"]
    assert any(
        item["caller"] == "a.py::target" and item["callee"] == "len"
        for item in unresolved
    )
