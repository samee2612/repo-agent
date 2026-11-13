"""Deterministic citation parser and ground-truth verifier."""

import logging
import re
from typing import Any

from app.models import Citation

logger = logging.getLogger(__name__)

# Matches citations such as `shop/auth.py:6-11` or `tests/test_auth.py:6`.
CITATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./:-])([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+):(\d+)(?:-(\d+))?(?!\d)"
)


def extract_citations(answer_text: str) -> list[Citation]:
    """Parse path:start-end and path:line citations from answer text."""
    seen: set[tuple[str, int, int]] = set()
    citations: list[Citation] = []

    for match in CITATION_PATTERN.finditer(answer_text):
        path = match.group(1).lstrip("./")
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line
        key = (path, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                path=path,
                start_line=start_line,
                end_line=end_line,
                verified=False,
            )
        )
    return citations


def collect_evidence_ranges(
    tool_name: str, result: dict[str, Any]
) -> list[tuple[str, int, int]]:
    """Extract (path, start_line, end_line) tuples from a successful tool result."""
    if not result.get("ok"):
        return []
    data = result.get("data") or {}
    ranges: list[tuple[str, int, int]] = []

    if tool_name == "search_code":
        for item in data.get("results", []):
            if "path" in item and "start_line" in item and "end_line" in item:
                ranges.append(
                    (item["path"], int(item["start_line"]), int(item["end_line"]))
                )
    elif tool_name == "read_file":
        if "path" in data and "start_line" in data and "end_line" in data:
            ranges.append(
                (data["path"], int(data["start_line"]), int(data["end_line"]))
            )
    elif tool_name == "list_symbols":
        for sym in data.get("symbols", []):
            if "path" in sym and "start_line" in sym and "end_line" in sym:
                ranges.append(
                    (sym["path"], int(sym["start_line"]), int(sym["end_line"]))
                )
    elif tool_name == "find_dependents":
        for dep in data.get("dependents", []):
            if "path" in dep and "start_line" in dep and "end_line" in dep:
                ranges.append(
                    (dep["path"], int(dep["start_line"]), int(dep["end_line"]))
                )

    return ranges


def verify_citations(
    answer_text: str, evidence_ranges: list[tuple[str, int, int]]
) -> tuple[list[Citation], bool]:
    """Verify extracted citations against tool-returned (path, start_line, end_line) ranges."""
    citations = extract_citations(answer_text)
    all_verified = True

    for citation in citations:
        if citation.start_line <= citation.end_line:
            for ev_path, ev_start, ev_end in evidence_ranges:
                norm_ev_path = ev_path.lstrip("./")
                if citation.path == norm_ev_path and (
                    citation.start_line <= ev_end and ev_start <= citation.end_line
                ):
                    citation.verified = True
                    break

        if not citation.verified:
            all_verified = False
            logger.warning(
                "Unverified citation in answer: %s:%d-%d",
                citation.path,
                citation.start_line,
                citation.end_line,
            )

    return citations, all_verified
