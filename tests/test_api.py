"""Tests for FastAPI endpoints using FakeLLMClient (no Gemini API calls)."""

from pathlib import Path
from fastapi.testclient import TestClient

from app.config import DEFAULT_SAMPLE_REPO
from app.graph import build_dependency_graph
from app.llm import FakeFunctionCall, FakeLLMClient, FakeResponse
from app.main import RepoState, create_app


def test_get_root_returns_html() -> None:
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "repo-agent" in res.text


def test_index_status_shape() -> None:
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    res = client.get("/index/status")
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"indexed", "files", "chunks", "edges"}
    assert body["indexed"] is False


def test_get_graph_empty_when_unindexed() -> None:
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    res = client.get("/graph")
    assert res.status_code == 200
    assert res.json() == {"nodes": [], "edges": []}


def test_get_graph_returns_expected_shape_for_sample_repo() -> None:
    app = create_app(llm_client=FakeLLMClient([]))
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO)
    app.state.repo_state = RepoState(
        indexed=True,
        files=5,
        chunks=9,
        edges=graph.number_of_edges(),
        repo_root=DEFAULT_SAMPLE_REPO,
        graph=graph,
        collection=None,
    )
    client = TestClient(app)
    res = client.get("/graph")
    assert res.status_code == 200
    body = res.json()
    assert "nodes" in body and "edges" in body
    node_ids = {n["id"] for n in body["nodes"]}
    assert "shop/auth.py" in node_ids
    assert "shop/auth.py::check_rate_limit" in node_ids
    assert any(
        e["source"] == "shop/orders.py::place_order"
        and e["target"] == "shop/auth.py::check_rate_limit"
        and e["kind"] == "calls"
        for e in body["edges"]
    )


def test_post_index_rejects_outside_allowed_root(tmp_path: Path) -> None:
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    res = client.post("/index", json={"path": str(tmp_path)})
    assert res.status_code == 400


def test_post_ask_with_fake_llm_client() -> None:
    fake_llm = FakeLLMClient(
        responses=[
            FakeResponse(
                function_calls=[
                    FakeFunctionCall(
                        name="read_file",
                        args={"path": "shop/auth.py", "start_line": 6, "end_line": 11},
                    )
                ]
            ),
            FakeResponse(
                text="check_rate_limit checks whether user_id is non-empty and within length limits (shop/auth.py:6-11)."
            ),
        ]
    )
    app = create_app(llm_client=fake_llm)
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO)
    app.state.repo_state = RepoState(
        indexed=True,
        files=5,
        chunks=9,
        edges=graph.number_of_edges(),
        repo_root=DEFAULT_SAMPLE_REPO,
        graph=graph,
        collection=None,
    )

    client = TestClient(app)
    res = client.post("/ask", json={"question": "How does check_rate_limit work?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["grounded"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["path"] == "shop/auth.py"
    assert body["citations"][0]["verified"] is True
    assert len(body["trace"]) == 1
    assert body["trace"][0]["tool"] == "read_file"
    assert set(body["highlight"].keys()) == {
        "nodes",
        "target_nodes",
        "edges",
        "unverified_nodes",
    }
    assert "shop/auth.py::check_rate_limit" in body["highlight"]["nodes"]
