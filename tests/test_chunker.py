"""Tests for AST-based code chunking."""

from app.indexer import extract_chunks_from_source


FIXTURE_SOURCE = '''"""Sample module."""

CONSTANT = 42


def helper(x: int) -> int:
    return x + 1


class Service:
    def run(self) -> str:
        return "ok"


async def fetch_data() -> dict:
    return {"ok": True}
'''


def test_extract_chunks_matches_top_level_functions_and_classes() -> None:
    chunks = extract_chunks_from_source(FIXTURE_SOURCE, "pkg/mod.py")
    assert [c.name for c in chunks] == ["helper", "Service", "fetch_data"]
    assert [c.kind for c in chunks] == ["function", "class", "async_function"]
    assert chunks[0].start_line == 6
    assert chunks[0].end_line == 7
    assert chunks[1].start_line == 10
    assert chunks[1].end_line == 12
    assert chunks[2].start_line == 15
    assert chunks[2].end_line == 16
    assert all(c.path == "pkg/mod.py" for c in chunks)
