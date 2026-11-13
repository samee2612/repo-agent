# repo-agent

An agentic question-answering service over a Python code repository with a browser-based chat UI. A Gemini-driven agent decides which repository tools to call, gathers evidence, and returns an answer where every citation is deterministically verified against tool results.

## Setup

```bash
cd repo-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Required environment variables

- `GEMINI_API_KEY`: Your Google Gemini API key.
- `GEMINI_MODEL`: The Gemini model name to use (for example `gemini-2.5-flash`). Not hardcoded; read at call time so tests run without an API key.

Optional environment variables:
- `ALLOWED_INDEX_ROOT`: Root directory permitted for `POST /index` (defaults to `sample_repo/`).
- `CHROMA_PERSIST_DIR`: Directory for persistent Chroma storage (defaults to `.chroma/`).
- `EVAL_SLEEP_SECONDS`: Delay between evaluation questions in `eval/run_eval.py` (defaults to `4`).

## Running the UI

1. Export the required environment variables and start the FastAPI server:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   export GEMINI_MODEL="gemini-2.5-flash"
   uvicorn app.main:app --reload
   ```
2. Open [http://localhost:8000](http://localhost:8000) in your browser.
3. Click **Index sample repo** in the top bar and wait for the status line to show the indexed file, chunk, and edge counts.
4. Ask a question in the chat box (press `Enter` to send, `Shift+Enter` for a newline).

### Example questions by category

- **lookup**: `Where is login defined and what parameters does it take?`
- **dependency**: `Which modules or functions depend on shop/auth.py::check_rate_limit?`
- **explain**: `How does place_order validate a user request before accepting an order?`
- **alias**: `What does throttle refer to and which function calls it?`
- **negative**: `How does the repository send confirmation emails via SMTP?`

## Docker

```bash
docker build -t repo-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" -e GEMINI_MODEL="$GEMINI_MODEL" repo-agent
```

## Split view

The browser UI uses a split-screen layout: chat on the left (45%) and an interactive dependency graph rendered with `vis-network` on the right (55%, stacked vertically below 900px):

- **What the graph shows**: All Python files (boxes) and top-level functions/classes (ellipses) extracted by `ast`, connected by `imports` (dashed) and `calls` (solid) edges from `GET /graph`.
- **What triggers highlights**: When an answer arrives (or when you click an earlier answer in the chat), non-matching nodes and edges dim. Cited/dependent nodes highlight in **green**, `find_dependents` target symbols highlight in **blue**, and any unverified citations highlight in **red with a dashed border**. Clicking a citation centers the graph on its node; clicking a node shows its details and citing questions; hovering a node highlights matching chat citations.
- **Deterministic provenance**: Highlights come strictly from deterministic tool results (`find_dependents`) and verified citations (`app/verifier.py` + `app/highlight.py`). The LLM never decides what gets highlighted.

## Design and limits

- **Why parsing is deterministic**: Code chunking and dependency graph construction use Python's standard-library `ast` module rather than an LLM. This guarantees reproducible symbol boundaries, exact line numbers, and zero-cost indexing without hallucinated edges.
- **Why the verifier exists**: LLMs can misquote line numbers or invent file paths. `app/verifier.py` deterministically parses `path:start-end` / `path:line` citations from the model's answer and checks each one against the actual `(path, start_line, end_line)` ranges returned by tools in that run. Unverified citations set `grounded: false` and are flagged in the UI without silently deleting text.
- **What the agent cannot answer**: The agent only inspects static source files and AST-resolved imports/calls. It cannot observe runtime behavior, dynamic imports (`importlib`, `exec`, `getattr`), external library internals, or questions outside the indexed repository.
- **LLM and rate limits**: Generation uses the `google-genai` SDK with Gemini function calling. The Gemini free tier enforces per-minute request limits; `GeminiClient` retries HTTP `429` and `503` with exponential backoff (up to 4 retries), and `eval/run_eval.py` sleeps between questions.
- **Security / demo scope**: The UI has no authentication and `POST /index` is restricted to `ALLOWED_INDEX_ROOT` to prevent arbitrary filesystem indexing. Do not expose this demo server directly to an untrusted network.
- **What eval numbers do and do not show**: `eval/run_eval.py` measures `recall@3` of `search_code`, citation validity rate, status counts, and per-category heuristic accuracy on the 10-question `sample_repo` benchmark. High scores show that the retrieval, graph, and citation pipeline work end-to-end on small synthetic code; they do not guarantee performance on large multi-language repositories or complex runtime bugs.

## Tests and Evaluation

Run unit and API tests (no Gemini API key required):
```bash
pytest
```

Run the evaluation suite (requires `GEMINI_API_KEY` and `GEMINI_MODEL`):
```bash
python eval/run_eval.py
```

### Verification status in build environment

In the sandboxed build environment, outbound PyPI access was blocked (`403 Forbidden`) and `GEMINI_API_KEY` / `GEMINI_MODEL` were not configured. As a result:
- Syntax compilation (`python3 -m py_compile`) and AST chunk extraction on `sample_repo/` were verified directly.
- Full `pytest` execution (requiring `pytest`, `networkx`, `chromadb`, `sentence-transformers`, `httpx`), `eval/run_eval.py`, and live browser UI verification could not be executed in this environment and should be run locally after `pip install -r requirements.txt`.
