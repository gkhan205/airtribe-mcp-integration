# Agent state machine

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	parse_jd(parse_jd)
	extract_requirements(extract_requirements)
	search_resumes(search_resumes)
	rank_candidates(rank_candidates)
	deep_analysis(deep_analysis)
	final_recommendation(final_recommendation)
	generate_report(generate_report)
	human_feedback(human_feedback)
	classify_feedback(classify_feedback)
	answer_question(answer_question)
	tools(tools<hr/><small><em>__tools = read_file, search_in_file, list_files, get_file_info, batch_process, get_screening_history, get_candidate_history, add_candidate_note, rag_search, extract_requirements, compare_candidates, generate_interview_questions, check_new_resumes</em></small>)
	__end__([<p>__end__</p>]):::last
	__start__ --> parse_jd;
	answer_question -.-> human_feedback;
	answer_question -.-> tools;
	classify_feedback -.-> __end__;
	classify_feedback -.-> answer_question;
	classify_feedback -.-> parse_jd;
	classify_feedback -.-> search_resumes;
	deep_analysis --> final_recommendation;
	extract_requirements --> search_resumes;
	final_recommendation --> generate_report;
	generate_report --> human_feedback;
	human_feedback --> classify_feedback;
	parse_jd --> extract_requirements;
	rank_candidates --> deep_analysis;
	search_resumes --> rank_candidates;
	tools --> answer_question;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Agent ↔ MCP servers

```mermaid
flowchart LR
    subgraph Agent["LangGraph agent (MCP client)"]
        parse_jd; deep_analysis; generate_report; tools["tools (Claude's tool calls)"]
    end
    client["MCPClient<br/>JSON-RPC 2.0"]
    subgraph FS["filesystem_mcp_server.py"]
        read_file; batch_process; write_file; watch_directory; search_in_file
    end
    subgraph DB["candidate_db_mcp_server.py"]
        save_screening; get_screening_history; add_candidate_note
    end
    parse_jd --> client; deep_analysis --> client; generate_report --> client; tools --> client
    client -- stdio --> FS
    client -- stdio --> DB
```
