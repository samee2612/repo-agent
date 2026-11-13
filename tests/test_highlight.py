"""Tests for deterministic graph highlight computation."""

from app.config import DEFAULT_SAMPLE_REPO
from app.graph import build_dependency_graph
from app.highlight import compute_highlight
from app.models import Citation


def test_verified_citation_highlights_file_and_symbol_nodes() -> None:
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO)
    citations = [
        Citation(path="shop/orders.py", start_line=7, end_line=18, verified=True)
    ]
    hl = compute_highlight(citations=citations, dependents_results=[], graph=graph)
    assert "shop/orders.py" in hl.nodes
    assert "shop/orders.py::place_order" in hl.nodes
    assert hl.unverified_nodes == []


def test_unverified_citation_is_excluded_from_nodes_and_added_to_unverified() -> None:
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO)
    citations = [
        Citation(path="shop/auth.py", start_line=6, end_line=11, verified=True),
        Citation(path="shop/payments.py", start_line=4, end_line=7, verified=False),
    ]
    hl = compute_highlight(citations=citations, dependents_results=[], graph=graph)
    assert "shop/auth.py::check_rate_limit" in hl.nodes
    assert "shop/payments.py::calculate_total" not in hl.nodes
    assert "shop/payments.py::calculate_total" in hl.unverified_nodes


def test_dependency_result_produces_target_dependents_and_edges() -> None:
    graph = build_dependency_graph(DEFAULT_SAMPLE_REPO)
    dependents_results = [
        {
            "symbol": "shop/auth.py::check_rate_limit",
            "dependents": [
                {
                    "source": "shop/orders.py::place_order",
                    "target": "shop/auth.py::check_rate_limit",
                    "relation": "calls",
                },
                {
                    "source": "shop/orders.py",
                    "target": "shop/auth.py::check_rate_limit",
                    "relation": "imports",
                },
            ],
        }
    ]
    hl = compute_highlight(
        citations=[], dependents_results=dependents_results, graph=graph
    )
    assert "shop/auth.py::check_rate_limit" in hl.target_nodes
    assert "shop/auth.py::check_rate_limit" in hl.nodes
    assert "shop/orders.py::place_order" in hl.nodes
    assert "shop/orders.py" in hl.nodes
    edge_tuples = {(e.source, e.target, e.kind) for e in hl.edges}
    assert ("shop/orders.py::place_order", "shop/auth.py::check_rate_limit", "calls") in edge_tuples
    assert ("shop/orders.py", "shop/auth.py::check_rate_limit", "imports") in edge_tuples
