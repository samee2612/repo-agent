"""Agentic question-answering loop with tool execution, limits, and citation verification."""

import time
from typing import Any

from app.config import AGENT_TIMEOUT_SECONDS, MAX_TOOL_ROUNDS
from app.highlight import compute_highlight
from app.llm import LLMClient
from app.models import Answer, TraceStep
from app.tools import RepoTools, get_gemini_tools
from app.verifier import collect_evidence_ranges, verify_citations

SYSTEM_PROMPT = (
    "You are a code repository question-answering assistant. "
    "Answer questions about the code repository using ONLY tool results. "
    "You MUST call tools to gather evidence before answering. "
    "Cite every claim using the exact format `path:start_line-end_line` "
    "(for example `shop/auth.py:6-11`). "
    "If the evidence from tool results is insufficient to answer the question, "
    "say so explicitly and do not guess or invent file paths or line numbers."
)


def _summarize_result(tool_name: str, result: dict[str, Any]) -> str:
    """Create a concise human-readable summary of a tool result for the trace."""
    if not result.get("ok"):
        return f"error: {result.get('error', 'unknown error')}"
    data = result.get("data") or {}
    if tool_name == "search_code":
        hits = data.get("results", [])
        top = ", ".join(f"{h['path']}:{h['start_line']}-{h['end_line']}" for h in hits[:3])
        return f"{len(hits)} hit(s)" + (f" [{top}]" if top else "")
    if tool_name == "find_dependents":
        deps = data.get("dependents", [])
        return f"{len(deps)} dependent(s)" + (f" ({data['note']})" if data.get("note") else "")
    if tool_name == "read_file":
        return f"{data.get('path')}:{data.get('start_line')}-{data.get('end_line')}"
    if tool_name == "list_symbols":
        syms = data.get("symbols", [])
        return f"{len(syms)} symbol(s) in {data.get('path')}"
    return "ok"


def _build_initial_history(question: str) -> list[Any]:
    """Build initial conversation history, using google.genai.types when available."""
    try:
        from google.genai import types

        return [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    except ImportError:
        return [{"role": "user", "text": question}]


def _build_function_response_content(
    responses: list[tuple[str, dict[str, Any]]]
) -> Any:
    """Build a Content block containing Part.from_function_response entries."""
    try:
        from google.genai import types

        parts = [
            types.Part.from_function_response(name=name, response=payload)
            for name, payload in responses
        ]
        return types.Content(role="user", parts=parts)
    except ImportError:
        return {"role": "tool", "responses": responses}


def _safe_get_gemini_tools() -> list[Any]:
    """Return Gemini tool declarations if google-genai is installed, else empty list."""
    try:
        return get_gemini_tools()
    except ImportError:
        return []


def run_agent(
    question: str,
    llm_client: LLMClient,
    repo_tools: RepoTools,
    max_rounds: int = MAX_TOOL_ROUNDS,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> Answer:
    """Run the tool-calling agent loop and return a verified Answer."""
    start_time = time.monotonic()
    history = _build_initial_history(question)
    tools = _safe_get_gemini_tools()

    trace: list[TraceStep] = []
    evidence_ranges: list[tuple[str, int, int]] = []
    dependents_results: list[dict[str, Any]] = []
    step_counter = 0

    for _round_idx in range(max_rounds):
        if time.monotonic() - start_time >= timeout_seconds:
            partial = (
                "Timed out before completing tool-assisted analysis. "
                f"Gathered {len(trace)} tool step(s)."
            )
            citations, grounded = verify_citations(partial, evidence_ranges)
            highlight = compute_highlight(citations, dependents_results, repo_tools.graph)
            return Answer(
                answer=partial,
                citations=citations,
                grounded=grounded,
                status="timeout",
                trace=trace,
                highlight=highlight,
            )

        response = llm_client.generate(
            history=history,
            tools=tools,
            system=SYSTEM_PROMPT,
        )

        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls:
            answer_text = (getattr(response, "text", None) or "").strip()
            if not answer_text:
                answer_text = "Insufficient evidence to answer the question."
            citations, grounded = verify_citations(answer_text, evidence_ranges)
            highlight = compute_highlight(citations, dependents_results, repo_tools.graph)
            return Answer(
                answer=answer_text,
                citations=citations,
                grounded=grounded,
                status="ok",
                trace=trace,
                highlight=highlight,
            )

        # Append model turn to history.
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            history.append(candidates[0].content)
        else:
            history.append({"role": "model", "function_calls": function_calls})

        # Execute each requested tool call.
        round_responses: list[tuple[str, dict[str, Any]]] = []
        for fc in function_calls:
            if time.monotonic() - start_time >= timeout_seconds:
                partial = (
                    "Timed out during tool execution. "
                    f"Gathered {len(trace)} tool step(s)."
                )
                citations, grounded = verify_citations(partial, evidence_ranges)
                highlight = compute_highlight(
                    citations, dependents_results, repo_tools.graph
                )
                return Answer(
                    answer=partial,
                    citations=citations,
                    grounded=grounded,
                    status="timeout",
                    trace=trace,
                    highlight=highlight,
                )

            step_counter += 1
            tool_name = fc.name
            tool_args = dict(getattr(fc, "args", None) or {})
            result = repo_tools.execute(tool_name, tool_args)

            if tool_name == "find_dependents" and result.get("ok") and result.get("data"):
                dependents_results.append(result["data"])

            evidence_ranges.extend(collect_evidence_ranges(tool_name, result))
            trace.append(
                TraceStep(
                    step=step_counter,
                    tool=tool_name,
                    input=tool_args,
                    result_summary=_summarize_result(tool_name, result),
                )
            )
            round_responses.append((tool_name, result))

        history.append(_build_function_response_content(round_responses))

    # Reached max_rounds without a final text answer.
    partial_lines = [
        "Reached maximum tool round limit before producing a final answer.",
        f"Executed {len(trace)} tool step(s).",
    ]
    if evidence_ranges:
        unique_ev = sorted(set(evidence_ranges))[:5]
        refs = ", ".join(f"{p}:{s}-{e}" for p, s, e in unique_ev)
        partial_lines.append(f"Partial evidence examined: {refs}.")
    partial_text = " ".join(partial_lines)
    citations, grounded = verify_citations(partial_text, evidence_ranges)
    highlight = compute_highlight(citations, dependents_results, repo_tools.graph)
    return Answer(
        answer=partial_text,
        citations=citations,
        grounded=grounded,
        status="step_limit",
        trace=trace,
        highlight=highlight,
    )
