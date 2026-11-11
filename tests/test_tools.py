"""
test_tools.py - the agent's tools and pure logic, no LLM needed (free and fast).
The file tools now run on the filesystem MCP server, so these tests also check the agent <-> MCP wiring.

Run:  pytest tests/test_tools.py -v
"""

from src.agent.nodes import explain_ranking_changes, parse_jd, requirements_to_must
from src.agent.tools import (
    build_must_list, get_all_tools, get_matcher, rag_search, read_resume_text, read_resumes_batch, resume_path,
)
from src.config.constants import JD_DIR


def test_agent_has_no_direct_file_system_access():
    """Part B: the agent and UI only reach files through MCP - nothing imports the old file tools."""
    import ast
    import pathlib
    for folder in ("src/agent", "src/ui"):
        for file in pathlib.Path(folder).glob("*.py"):
            tree = ast.parse(file.read_text())
            imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            assert not any("file_system" in (name or "") for name in imported), file


def test_read_resume_text_goes_through_mcp():
    rid = next(iter(get_matcher().rag.resumes))
    assert len(read_resume_text(rid)) > 100


def test_read_resumes_batch_reads_all_in_one_call():
    rids = list(get_matcher().rag.resumes)[:10]
    texts = read_resumes_batch(rids)
    assert set(texts) == set(rids) and all(texts.values())


def test_tools_for_claude_include_mcp_and_agent_tools():
    names = {t.name for t in get_all_tools()}
    assert {"read_file", "search_in_file", "batch_process", "get_candidate_history"} <= names   # from MCP servers
    assert {"rag_search", "compare_candidates", "check_new_resumes"} <= names                   # the agent's own
    assert "write_file" not in names  # Claude does not get raw write access


def test_mcp_read_file_tool_blocks_path_traversal():
    read_file = next(t for t in get_all_tools() if t.name == "read_file")
    result = read_file.invoke({"path": "../.env"})
    assert result["success"] is False and result["code"] == "ACCESS_DENIED"
    assert "ANTHROPIC" not in str(result)


def test_mcp_search_in_file_finds_keyword():
    rid = next(r for r, m in get_matcher().rag.resumes.items() if "Excel" in m["skills"])
    search = next(t for t in get_all_tools() if t.name == "search_in_file")
    assert search.invoke({"path": resume_path(rid), "keyword": "excel"})["match_count"] > 0


def test_rag_search_must_filter():
    result = rag_search.invoke({"query": "IT systems administrator", "k": 10, "must": ["Linux"]})
    resumes = get_matcher().rag.resumes
    assert result["candidates"]
    assert all("Linux" in resumes[c["candidate_id"]]["skills"] for c in result["candidates"])


def test_unknown_skill_is_skipped_not_crashing():
    valid, skipped = build_must_list("Title\nSome job", ["Linux", "Kubernetes:2"], [])
    assert valid == ["Linux"] and skipped == ["Kubernetes:2"]


def test_ignored_must_removes_jd_requirement():
    jd = (JD_DIR / "it_systems_admin.txt").read_text()  # its must-haves: 3+ years experience, Networking
    valid, _ = build_must_list(jd, [], ["Networking"])
    assert valid == ["years:3"]


def test_requirements_to_must():
    assert requirements_to_must({"must_have": ["SQL", "3+ years experience"], "min_years": 3}) == ["SQL", "years:3"]


def test_explain_ranking_changes():
    def m(name, score):
        return {"candidate_name": name, "resume_path": f"data/resumes/{name.lower()}.pdf", "match_score": score}
    previous = [m("Ann", 80), m("Bob", 70), m("Cat", 60)]
    current = [m("Bob", 75), m("Ann", 72), m("Dan", 50)]
    changes = " ".join(explain_ranking_changes(previous, current, [{"candidate_name": "Cat", "reasons": ["missing must-have skill Linux"]}]))
    assert "Bob moved up #2 → #1 (score 70 → 75" in changes
    assert "Ann moved down #1 → #2 (score 80 → 72" in changes
    assert "Dan entered" in changes
    assert "Cat (was #3) dropped out: missing must-have skill Linux" in changes


def test_parse_jd_reads_file_by_name():
    state = parse_jd({"job_description": "teacher"})
    assert "Teacher" in state["job_description"]
