"""Tests for deterministic citation verification."""

from app.verifier import verify_citations


def test_verifier_valid_citation() -> None:
    answer = "Rate limits are checked in shop/auth.py:6-11."
    evidence = [("shop/auth.py", 6, 11)]
    citations, grounded = verify_citations(answer, evidence)
    assert grounded is True
    assert len(citations) == 1
    assert citations[0].verified is True


def test_verifier_overlapping_citation() -> None:
    answer = "See shop/orders.py:8-10 and shop/orders.py:12."
    evidence = [("shop/orders.py", 7, 18)]
    citations, grounded = verify_citations(answer, evidence)
    assert grounded is True
    assert len(citations) == 2
    assert all(c.verified for c in citations)


def test_verifier_invalid_citation_marks_ungrounded() -> None:
    answer = "Check shop/auth.py:6-11 and shop/fake.py:100-110."
    evidence = [("shop/auth.py", 6, 11)]
    citations, grounded = verify_citations(answer, evidence)
    assert grounded is False
    assert len(citations) == 2
    assert citations[0].verified is True
    assert citations[1].verified is False
