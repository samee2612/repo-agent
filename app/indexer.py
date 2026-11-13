"""Deterministic AST-based repository indexer and Chroma vector store manager."""

import ast
from pathlib import Path
from typing import Any

from app.config import ALLOWED_INDEX_ROOT, CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from app.models import Chunk

_EMBEDDER: Any = None


def get_embedder() -> Any:
    """Lazily load and cache the SentenceTransformer embedding model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER


def validate_index_path(
    path_str: str | None, allowed_root: Path = ALLOWED_INDEX_ROOT
) -> Path:
    """Validate that the requested index path is inside allowed_root."""
    allowed_resolved = allowed_root.resolve()
    if not path_str:
        target = allowed_resolved
    else:
        target = Path(path_str).expanduser().resolve()

    if not target.is_relative_to(allowed_resolved):
        raise ValueError(
            f"Path '{target}' is outside ALLOWED_INDEX_ROOT '{allowed_resolved}'."
        )
    if not target.exists() or not target.is_dir():
        raise ValueError(f"Path '{target}' does not exist or is not a directory.")
    return target


def iter_python_files(repo_root: Path) -> list[Path]:
    """Walk repo_root and return sorted .py files, skipping hidden dirs and __pycache__."""
    py_files: list[Path] = []
    for path in sorted(repo_root.rglob("*.py")):
        rel_parts = path.relative_to(repo_root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
            continue
        if path.is_file():
            py_files.append(path)
    return py_files


def extract_chunks_from_source(source: str, rel_path: str) -> list[Chunk]:
    """Parse Python source with ast and extract top-level functions and classes."""
    tree = ast.parse(source, filename=rel_path)
    lines = source.splitlines()
    chunks: list[Chunk] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            kind = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue

        start_line = node.lineno
        end_line = node.end_lineno or node.lineno
        snippet = "\n".join(lines[start_line - 1 : end_line])
        chunks.append(
            Chunk(
                path=rel_path,
                name=node.name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                source=snippet,
            )
        )

    return chunks


def extract_repo_chunks(repo_root: Path) -> tuple[list[Path], list[Chunk]]:
    """Extract all top-level function and class chunks from Python files in repo_root."""
    files = iter_python_files(repo_root)
    all_chunks: list[Chunk] = []
    for file_path in files:
        rel_path = file_path.relative_to(repo_root).as_posix()
        source = file_path.read_text(encoding="utf-8")
        all_chunks.extend(extract_chunks_from_source(source, rel_path))
    return files, all_chunks


def get_chroma_collection(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = "repo_chunks",
    reset: bool = False,
) -> Any:
    """Return a persistent Chroma collection, optionally resetting it first."""
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks_to_chroma(
    chunks: list[Chunk],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = "repo_chunks",
    embedder: Any = None,
) -> int:
    """Embed chunks with sentence-transformers and store them in Chroma."""
    collection = get_chroma_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        reset=True,
    )
    if not chunks:
        return 0

    model = embedder or get_embedder()
    texts = [c.source for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embedding_list = [
        emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings
    ]

    ids = [f"{c.path}::{c.name}:{c.start_line}" for c in chunks]
    metadatas = [
        {
            "path": c.path,
            "name": c.name,
            "kind": c.kind,
            "start_line": c.start_line,
            "end_line": c.end_line,
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embedding_list,
        metadatas=metadatas,
    )
    return len(chunks)
