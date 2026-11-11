"""Entry point for the Resume RAG & job matching app.

Commands:
    python -m src.rag.cli build     [resumes/] [--chunker fixed]   Part A: chunk, embed and index resumes (FAISS)
    python -m src.rag.cli query     "text" [-k 5]                  raw semantic search over resume chunks
    python -m src.rag.cli show      data/resumes/<file>.pdf             extracted metadata + chunks for one resume
    python -m src.rag.cli match     data/job_descriptions/<jd>.txt ...  Part B: rank candidates for job description(s)
    python -m src.rag.cli evaluate  [--runs 50]                    retrieval accuracy + latency metrics

Run `python -m src.rag.cli <command> -h` for a command's options.
To (re)download the sample resumes: python scripts/fetch_resumes.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]

    if cmd in ("build", "query", "show"):
        from src.rag.resume_rag import cli
        cli([cmd, *rest])
    elif cmd == "match":
        from src.rag.job_matcher import cli
        cli(rest)
    elif cmd == "evaluate":
        from src.rag.evaluation import cli
        cli(rest)
    else:
        print(f"Unknown command: {cmd}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
