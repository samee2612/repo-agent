"""FastAPI application serving the repo-agent API and browser chat UI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.agent import run_agent
from app.config import ALLOWED_INDEX_ROOT, CHROMA_PERSIST_DIR
from app.graph import build_dependency_graph
from app.indexer import (
    extract_repo_chunks,
    get_chroma_collection,
    index_chunks_to_chroma,
    validate_index_path,
)
from app.llm import GeminiClient, LLMClient
from app.models import Answer, AskRequest, IndexRequest, IndexStatus
from app.tools import RepoTools

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class RepoState:
    """In-memory state for the currently indexed repository."""

    indexed: bool = False
    files: int = 0
    chunks: int = 0
    edges: int = 0
    repo_root: Path = ALLOWED_INDEX_ROOT
    graph: Any = None
    collection: Any = None
    embedder: Any = None


def create_app(llm_client: LLMClient | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(title="repo-agent", version="0.1.0")
    application.state.repo_state = RepoState()
    application.state.llm_client = llm_client or GeminiClient()

    @application.get("/")
    def serve_ui() -> FileResponse:
        """Serve the single-file browser chat UI."""
        index_html = STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="UI file not found.")
        return FileResponse(index_html, media_type="text/html")

    @application.get("/index/status", response_model=IndexStatus)
    def get_index_status() -> IndexStatus:
        """Return the current repository index status and counts."""
        state: RepoState = application.state.repo_state
        return IndexStatus(
            indexed=state.indexed,
            files=state.files,
            chunks=state.chunks,
            edges=state.edges,
        )

    @application.post("/index", response_model=IndexStatus)
    def index_repository(payload: IndexRequest | None = Body(default=None)) -> IndexStatus:
        """Index a repository directory inside ALLOWED_INDEX_ROOT."""
        requested_path = payload.path if payload else None
        try:
            repo_root = validate_index_path(requested_path, ALLOWED_INDEX_ROOT)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        state: RepoState = application.state.repo_state
        try:
            files, chunks = extract_repo_chunks(repo_root)
            chunk_count = index_chunks_to_chroma(
                chunks=chunks,
                persist_dir=CHROMA_PERSIST_DIR,
                embedder=state.embedder,
            )
            collection = get_chroma_collection(persist_dir=CHROMA_PERSIST_DIR, reset=False)
            graph = build_dependency_graph(repo_root, chunks=chunks)
        except Exception as exc:
            import logging

            logging.exception("Indexing failed")
            raise HTTPException(status_code=500, detail=f"Indexing error: {exc}") from exc

        state.indexed = True
        state.files = len(files)
        state.chunks = chunk_count
        state.edges = graph.number_of_edges()
        state.repo_root = repo_root
        state.graph = graph
        state.collection = collection

        return IndexStatus(
            indexed=state.indexed,
            files=state.files,
            chunks=state.chunks,
            edges=state.edges,
        )

    @application.post("/ask", response_model=Answer)
    def ask_question(payload: AskRequest) -> Answer:
        """Run the agentic QA loop for a question over the indexed repository."""
        state: RepoState = application.state.repo_state
        if not state.indexed:
            raise HTTPException(
                status_code=400,
                detail="Repository is not indexed yet. Click 'Index sample repo' first.",
            )

        repo_tools = RepoTools(
            repo_root=state.repo_root,
            graph=state.graph,
            collection=state.collection,
            embedder=state.embedder,
        )
        client: LLMClient = application.state.llm_client
        try:
            return run_agent(
                question=payload.question,
                llm_client=client,
                repo_tools=repo_tools,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return application


app = create_app()
