"""Repository inspection tools, Pydantic input models, and Gemini FunctionDeclarations."""

from pathlib import Path
from typing import Any
import networkx as nx
from pydantic import BaseModel, Field

from app.config import MAX_READ_LINES
from app.indexer import extract_chunks_from_source, get_embedder
from app.models import ToolResult


class SearchCodeInput(BaseModel):
    """Input schema for search_code."""

    query: str = Field(..., min_length=1, description="Natural language or code search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of chunks to return.")


class FindDependentsInput(BaseModel):
    """Input schema for find_dependents."""

    symbol: str = Field(
        ...,
        min_length=1,
        description="Canonical symbol identifier in the form 'path::name', e.g. 'shop/auth.py::check_rate_limit'.",
    )


class ReadFileInput(BaseModel):
    """Input schema for read_file."""

    path: str = Field(..., min_length=1, description="Relative path to a file inside the repository.")
    start_line: int | None = Field(default=None, ge=1, description="1-based start line number.")
    end_line: int | None = Field(default=None, ge=1, description="1-based end line number.")


class ListSymbolsInput(BaseModel):
    """Input schema for list_symbols."""

    path: str = Field(..., min_length=1, description="Relative path to a Python file in the repository.")


class RepoTools:
    """Stateful tool executor bound to an indexed repository, graph, and Chroma collection."""

    def __init__(
        self,
        repo_root: Path,
        graph: nx.DiGraph | None = None,
        collection: Any = None,
        embedder: Any = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.graph = graph
        self.collection = collection
        self.embedder = embedder

    def _resolve_safe_path(self, rel_path: str) -> Path:
        """Resolve rel_path inside repo_root, raising ValueError if it escapes."""
        candidate = (self.repo_root / rel_path).resolve()
        if not candidate.is_relative_to(self.repo_root):
            raise ValueError(f"Path '{rel_path}' escapes repository root.")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"File '{rel_path}' does not exist in repository.")
        return candidate

    def search_code(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Embed the query, search Chroma, and return matching code chunks."""
        if self.collection is None:
            return {"ok": False, "error": "Repository is not indexed yet."}
        model = self.embedder or get_embedder()
        embedding = model.encode([query], normalize_embeddings=True)
        query_vec = (
            embedding[0].tolist() if hasattr(embedding[0], "tolist") else list(embedding[0])
        )
        raw = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = round(1.0 - float(dist), 4) if dist is not None else None
            item = ToolResult(
                path=meta["path"],
                name=meta["name"],
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                score=score,
                snippet=doc,
            )
            results.append(item.model_dump())

        return {"ok": True, "data": {"results": results}}

    def find_dependents(self, symbol: str) -> dict[str, Any]:
        """Return files and symbols with an edge into symbol, or an empty list with a note."""
        if self.graph is None:
            return {"ok": False, "error": "Dependency graph is not built yet."}
        if symbol not in self.graph:
            return {
                "ok": True,
                "data": {
                    "symbol": symbol,
                    "dependents": [],
                    "note": f"Unknown symbol '{symbol}'. No dependents found.",
                },
            }

        dependents: list[dict[str, Any]] = []
        for source, target, edge_data in self.graph.in_edges(symbol, data=True):
            node_data = self.graph.nodes[source]
            entry: dict[str, Any] = {
                "source": source,
                "target": target,
                "source_type": node_data.get("type", "unknown"),
                "path": node_data.get("path", source),
                "relation": edge_data.get("relation", "depends_on"),
                "edge_path": f"{source} -[{edge_data.get('relation', 'depends_on')}]-> {target}",
            }
            if "start_line" in node_data and "end_line" in node_data:
                entry["start_line"] = int(node_data["start_line"])
                entry["end_line"] = int(node_data["end_line"])
            elif "line" in edge_data:
                entry["start_line"] = int(edge_data["line"])
                entry["end_line"] = int(edge_data["line"])
            dependents.append(entry)

        note = None if dependents else f"Symbol '{symbol}' has no incoming dependents."
        return {
            "ok": True,
            "data": {"symbol": symbol, "dependents": dependents, "note": note},
        }

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read requested lines from a repository file with 1-based line numbers."""
        try:
            target = self._resolve_safe_path(path)
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}

        lines = target.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        start = start_line if start_line is not None else 1
        end = end_line if end_line is not None else min(total_lines, start + MAX_READ_LINES - 1)

        if start > end:
            return {"ok": False, "error": f"start_line ({start}) cannot exceed end_line ({end})."}
        if total_lines == 0:
            return {
                "ok": True,
                "data": {"path": target.relative_to(self.repo_root).as_posix(), "start_line": 1, "end_line": 1, "content": ""},
            }

        start = max(1, min(start, total_lines))
        end = max(start, min(end, total_lines, start + MAX_READ_LINES - 1))

        numbered = [f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)]
        rel_posix = target.relative_to(self.repo_root).as_posix()
        return {
            "ok": True,
            "data": {
                "path": rel_posix,
                "start_line": start,
                "end_line": end,
                "total_lines": total_lines,
                "content": "\n".join(numbered),
            },
        }

    def list_symbols(self, path: str) -> dict[str, Any]:
        """List all top-level functions and classes in a Python file with line ranges."""
        try:
            target = self._resolve_safe_path(path)
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}

        rel_posix = target.relative_to(self.repo_root).as_posix()
        try:
            source = target.read_text(encoding="utf-8")
            chunks = extract_chunks_from_source(source, rel_posix)
        except SyntaxError as exc:
            return {"ok": False, "error": f"Failed to parse '{rel_posix}': {exc}"}

        symbols = [
            {
                "path": c.path,
                "name": c.name,
                "symbol_id": c.symbol_id,
                "kind": c.kind,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ]
        return {"ok": True, "data": {"path": rel_posix, "symbols": symbols}}

    def execute(self, name: str, raw_args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate arguments with Pydantic and invoke the named tool safely."""
        args = raw_args or {}
        try:
            if name == "search_code":
                parsed = SearchCodeInput.model_validate(args)
                return self.search_code(query=parsed.query, top_k=parsed.top_k)
            if name == "find_dependents":
                parsed = FindDependentsInput.model_validate(args)
                return self.find_dependents(symbol=parsed.symbol)
            if name == "read_file":
                parsed = ReadFileInput.model_validate(args)
                return self.read_file(
                    path=parsed.path,
                    start_line=parsed.start_line,
                    end_line=parsed.end_line,
                )
            if name == "list_symbols":
                parsed = ListSymbolsInput.model_validate(args)
                return self.list_symbols(path=parsed.path)
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def get_gemini_tools() -> list[Any]:
    """Return a list containing one google.genai.types.Tool with all FunctionDeclarations."""
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name="search_code",
            description="Semantic search over indexed Python functions and classes. Returns matching chunks with file path, symbol name, line range, score, and snippet.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="Natural language or code search query.",
                    ),
                    "top_k": types.Schema(
                        type="INTEGER",
                        description="Maximum number of chunks to return (default 5).",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="find_dependents",
            description="Find all files and symbols that import or call a given symbol (format: 'path::name').",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "symbol": types.Schema(
                        type="STRING",
                        description="Canonical symbol identifier in the form 'path::name', e.g. 'shop/auth.py::check_rate_limit'.",
                    ),
                },
                required=["symbol"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read lines from a file in the repository with 1-based line numbers.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING",
                        description="Relative path to a file inside the repository.",
                    ),
                    "start_line": types.Schema(
                        type="INTEGER",
                        description="Optional 1-based start line number.",
                    ),
                    "end_line": types.Schema(
                        type="INTEGER",
                        description="Optional 1-based end line number.",
                    ),
                },
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_symbols",
            description="List all top-level functions and classes in a Python file with their line ranges.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING",
                        description="Relative path to a Python file in the repository.",
                    ),
                },
                required=["path"],
            ),
        ),
    ]
    return [types.Tool(function_declarations=declarations)]
