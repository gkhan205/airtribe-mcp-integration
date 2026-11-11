# Test scenarios

## 1. MCP server and client (offline, no LLM, about 1 minute)

| File | What it shows |
|---|---|
| `tests/test_mcp_server.py` (26 tests) | Tool discovery and resource discovery; every Milestone 1 tool; status codes for bad paths (`../.env`, `/etc/passwd`, missing, folder, wrong type, too large, empty); JSON-RPC codes for bad resources (`-32002`, `-32602`); resources return data; `batch_process` reads a whole folder, keeps going when one file fails, enforces limits; `watch_directory` notices a file dropped in **while it is waiting** and does not report it twice; configuration layers (defaults → JSON → env); **raw JSON-RPC 2.0 over stdio** against a real subprocess |
| `tests/test_mcp_client.py` (8 tests) | The client starts and connects to **both** servers; tools are routed to the right server; tool errors become `{"success": False, "code": ...}`; resources work across servers; MCP tools become LangChain tools; the candidate database saves screenings and notes; progress notifications arrive while `watch_directory` waits |
| `tests/test_tools.py` (12 tests) | Agent ↔ MCP wiring: the agent and UI do **not import** the old file tools at all; resumes are read via MCP (one at a time and as a batch); Claude gets the MCP tools plus the agent's tools but not `write_file`; path traversal is blocked through the agent's tool; plus the Milestone 3 logic tests (RAG filters, ranking changes) |

Run them with: `pytest tests/test_mcp_server.py tests/test_mcp_client.py tests/test_tools.py -v`

## 2. Conversation flows (call Claude)

Each scenario is an automated test in `tests/test_scenarios.py` (needs `ANTHROPIC_API_KEY`) (run with `pytest tests/test_scenarios.py -v`).
The tests check the agent **state** after the conversation, not the exact wording, because LLM text changes between runs.

| # | Conversation | What the agent does | What the test checks |
|---|---|---|---|
| 1 | *"Find me candidates with SQL and 3+ years experience"* | LLM extracts must-haves (SQL, 3 years) and turns them into hard filters for the RAG search | Every shortlisted candidate has SQL and ≥ 3 years of experience |
| 2 | `it_systems_admin` → *"Compare the top 3 matches side by side"* | Classified as a question; Claude calls `compare_candidates` and shows a table plus a verdict | `compare_candidates` was called with exactly the top 3 ids |
| 3 | `it_systems_admin` → *"Why did A rank higher than B?"* | Explains the gap using the score breakdown (semantic / skills / keyword / qualification), skills and experience | The answer names both candidates |
| 4 | `it_systems_admin` → *"Make Linux a must-have"* | Classified as a refinement; re-runs all 3 rounds and lists who moved, entered or dropped out, and why | `Linux` is in `extra_must`, the old shortlist is kept, everyone left has Linux, changes are explained |
| 5 | `it_systems_admin` (multi-round) | Round 1: 100 resumes → top 10. Round 2: LLM analysis of each. Round 3: hire / borderline / no-hire | 100 screened, ≤ 10 shortlisted, analysis + decision for each, borderline ones have suggestions, report file exists |
| 6 | `it_systems_admin` → *"Give me interview questions for the top candidate"* | Claude calls `generate_interview_questions`, which targets the candidate's gaps | The tool was called and returned 5 questions |
| 7 | `it_systems_admin` → *"Let's look at a different role: data/job_descriptions/teacher.txt"* → *"done, thanks"* | Classified as `new_job`: a fresh screening for Teacher, then `done` ends the graph | The JD switched to Teacher, the old shortlist was cleared, the graph reached END |
| 8 | `it_systems_admin` → *"Look in the top candidate's resume: does it mention Windows?"* | Claude uses the **MCP file tools** (`search_in_file` / `read_file`) with the candidate's `resume_path` | An MCP file tool was called and succeeded |
| 9 | `it_systems_admin` → *"Check for new resumes for the next 30 seconds"*, and a resume is dropped into `data/resumes/` 5 s later | `check_new_resumes` → MCP `watch_directory` returns as soon as the file appears → MCP `batch_process` reads it → Claude analyses it for the job | The new file was reported and analysed with a fit score |
| 10 | `it_systems_admin` → *"Add a note for <top candidate>: available to start in January"* | **Second MCP server**: Claude calls `add_candidate_note` on `candidate_db`. The screening itself was saved there automatically by `generate_report` | The note is stored in the database; the screening appears in `get_screening_history` |

## Example run (abridged, from a real session)

```
You: it_systems_admin
  📄 Reading the job description
  🧩 Extracting must-have vs nice-to-have requirements (LLM)
  🔎 Round 1: hybrid RAG search over all resumes → 100 resumes, 10 shortlisted, 54 filtered out
  📊 Ranking candidates and comparing with the previous ranking
  🧠 Round 2: deep analysis of the top 10 (LLM, in parallel)
  ⚖️ Round 3: hire / borderline / no-hire decisions (LLM)
  📝 Writing the match report
Agent: IT Systems Administrator: screened 100 resumes → top 10 → 1 hire, 1 borderline, 8 no-hire
  | 1 | Ines Brown | 79 | 6/10 | 🟡 borderline |
  | 2 | Elena Reed | 78 | 7/10 | ✅ hire |
  ...

You: Make Linux a must-have
  🤔 Understanding your request (LLM) → refine (add ['Linux'])
  🔎 Round 1: ... → 100 resumes, 5 shortlisted, 63 filtered out
Agent: What changed:
  - ⬆️ Amara Rahman moved up #3 → #2 (score unchanged at 58; candidates above were filtered out)
  - 🆕 Leila Aziz entered at #5 (score 29)
  - ❌ Elena Reed (was #2) dropped out: missing must-have skill Linux
  ...

You: Now drop the Linux requirement and require 10+ years of experience instead
  🤔 Understanding your request (LLM) → refine (add ['years:10']) (remove ['Linux'])
Agent: What changed:
  - 🆕 Elena Reed entered at #1 (score 78)
  - ❌ Ines Brown (was #1) dropped out: 4.6 yrs experience < 10 required
  ...
```
