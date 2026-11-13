"""Deterministic graph highlight computation from tool results and verified citations."""

from typing import Any
import networkx as nx

from app.models import Citation, HighlightData, HighlightEdge


def _map_citation_to_nodes(
    citation: Citation, graph: nx.DiGraph | None
) -> list[str]:
    """Map a citation's path and line range to matching graph node ids."""
    if graph is None:
        return [citation.path]

    matched_symbols: list[str] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "symbol" and data.get("path") == citation.path:
            start = int(data.get("start_line", 0))
            end = int(data.get("end_line", 0))
            if citation.start_line <= end and start <= citation.end_line:
                matched_symbols.append(node_id)

    result: list[str] = []
    if citation.path in graph:
        result.append(citation.path)
    result.extend(matched_symbols)

    if not result:
        result.append(citation.path)
    return result


def compute_highlight(
    citations: list[Citation],
    dependents_results: list[dict[str, Any]],
    graph: nx.DiGraph | None,
) -> HighlightData:
    """Compute highlighted nodes, target nodes, edges, and unverified nodes deterministically."""
    node_set: set[str] = set()
    ordered_nodes: list[str] = []
    target_set: set[str] = set()
    ordered_targets: list[str] = []
    unverified_set: set[str] = set()
    ordered_unverified: list[str] = []
    edge_keys: set[tuple[str, str, str]] = set()
    ordered_edges: list[HighlightEdge] = []

    def add_node(nid: str) -> None:
        if nid not in node_set:
            node_set.add(nid)
            ordered_nodes.append(nid)

    def add_target(nid: str) -> None:
        add_node(nid)
        if nid not in target_set:
            target_set.add(nid)
            ordered_targets.append(nid)

    def add_edge(src: str, tgt: str, kind: str) -> None:
        key = (src, tgt, kind)
        if key not in edge_keys:
            edge_keys.add(key)
            ordered_edges.append(HighlightEdge(source=src, target=tgt, kind=kind))

    # 1. Process verified and unverified citations.
    for cit in citations:
        mapped = _map_citation_to_nodes(cit, graph)
        if cit.verified:
            for nid in mapped:
                add_node(nid)
        else:
            for nid in mapped:
                if nid not in unverified_set:
                    unverified_set.add(nid)
                    ordered_unverified.append(nid)

    # 2. Process find_dependents tool results.
    for dep_data in dependents_results:
        target = dep_data.get("symbol")
        if target and (graph is None or target in graph):
            add_target(target)
        for dep in dep_data.get("dependents", []):
            src = dep.get("source")
            tgt = dep.get("target") or target
            rel = dep.get("relation", "calls")
            if src:
                add_node(src)
            if src and tgt:
                add_edge(src, tgt, rel)

    # 3. Include any `calls` edge between two highlighted nodes.
    if graph is not None:
        for src, tgt, edge_data in graph.edges(data=True):
            if (
                edge_data.get("relation") == "calls"
                and src in node_set
                and tgt in node_set
            ):
                add_edge(src, tgt, "calls")

    return HighlightData(
        nodes=ordered_nodes,
        target_nodes=ordered_targets,
        edges=ordered_edges,
        unverified_nodes=ordered_unverified,
    )
