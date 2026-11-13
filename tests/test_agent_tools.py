"""Tests for repository tools and agent error handling."""

from pathlib import Path
from app.graph import build_dependency_graph
from app.tools import RepoTools


def test_read_file_rejects_parent_traversal(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    tools = RepoTools(repo_root=tmp_path)

    res = tools.read_file("../etc/passwd")
    assert res["ok"] is False
    assert "escapes" in res["error"]


def test_find_dependents_unknown_symbol_returns_empty_with_note(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    graph = build_dependency_graph(tmp_path)
    tools = RepoTools(repo_root=tmp_path, graph=graph)

    res = tools.find_dependents("mod.py::does_not_exist")
    assert res["ok"] is True
    assert res["data"]["dependents"] == []
    assert res["data"]["note"]
