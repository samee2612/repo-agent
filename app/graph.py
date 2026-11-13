"""Deterministic AST-based dependency graph builder using NetworkX."""

import ast
from pathlib import Path
import networkx as nx

from app.indexer import iter_python_files
from app.models import Chunk


def _module_name_from_rel_path(rel_path: str) -> str:
    """Convert a relative .py path to a dotted Python module name."""
    pure = Path(rel_path)
    parts = list(pure.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _extract_call_names(node: ast.AST) -> list[tuple[str, int]]:
    """Return (callee_name, line_number) pairs for calls inside an AST node."""
    calls: list[tuple[str, int]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append((child.func.id, child.lineno))
            elif isinstance(child.func, ast.Attribute) and isinstance(
                child.func.value, ast.Name
            ):
                calls.append((f"{child.func.value.id}.{child.func.attr}", child.lineno))
    return calls


def build_dependency_graph(
    repo_root: Path, chunks: list[Chunk] | None = None
) -> nx.DiGraph:
    """Build a NetworkX DiGraph of files, symbols, imports, and resolved calls."""
    graph = nx.DiGraph()
    graph.graph["unresolved"] = []

    py_files = iter_python_files(repo_root)
    module_to_path: dict[str, str] = {}
    parsed_trees: dict[str, ast.Module] = {}
    symbols_by_file: dict[str, dict[str, str]] = {}
    known_symbols: set[str] = set()

    for file_path in py_files:
        rel_path = file_path.relative_to(repo_root).as_posix()
        graph.add_node(rel_path, type="file", path=rel_path)
        mod_name = _module_name_from_rel_path(rel_path)
        if mod_name:
            module_to_path[mod_name] = rel_path
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        parsed_trees[rel_path] = tree
        symbols_by_file[rel_path] = {}

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sym_id = f"{rel_path}::{node.name}"
                kind = (
                    "class"
                    if isinstance(node, ast.ClassDef)
                    else (
                        "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    )
                )
                graph.add_node(
                    sym_id,
                    type="symbol",
                    path=rel_path,
                    name=node.name,
                    kind=kind,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
                symbols_by_file[rel_path][node.name] = sym_id
                known_symbols.add(sym_id)

    # Second pass: resolve imports, aliases, and calls per file.
    for rel_path, tree in parsed_trees.items():
        local_table: dict[str, str] = dict(symbols_by_file[rel_path])
        module_aliases: dict[str, str] = {}

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                target_path = module_to_path.get(node.module)
                if not target_path:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    target_sym = f"{target_path}::{alias.name}"
                    local_name = alias.asname or alias.name
                    if target_sym in known_symbols:
                        local_table[local_name] = target_sym
                        graph.add_edge(
                            rel_path,
                            target_sym,
                            relation="imports",
                            line=node.lineno,
                        )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target_path = module_to_path.get(alias.name)
                    if target_path:
                        local_name = alias.asname or alias.name.split(".")[0]
                        module_aliases[local_name] = target_path

            elif isinstance(node, ast.Assign):
                # Track simple module-level aliases like `throttle = check_rate_limit`.
                if isinstance(node.value, ast.Name) and node.value.id in local_table:
                    resolved = local_table[node.value.id]
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            local_table[target.id] = resolved

        # Resolve calls inside top-level functions/methods.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                caller_id = f"{rel_path}::{node.name}"
                for callee_name, line in _extract_call_names(node):
                    target_sym = None
                    if callee_name in local_table:
                        target_sym = local_table[callee_name]
                    elif "." in callee_name:
                        mod_alias, attr_name = callee_name.split(".", 1)
                        if mod_alias in module_aliases:
                            candidate = f"{module_aliases[mod_alias]}::{attr_name}"
                            if candidate in known_symbols:
                                target_sym = candidate

                    if target_sym and target_sym in known_symbols:
                        graph.add_edge(
                            caller_id,
                            target_sym,
                            relation="calls",
                            line=line,
                        )
                    else:
                        graph.graph["unresolved"].append(
                            {
                                "caller": caller_id,
                                "callee": callee_name,
                                "line": line,
                            }
                        )

    return graph
