"""Retrieval accuracy and latency evaluation against data/job_descriptions/ground_truth.json.

Usage (from the project root):
    python -m src.rag.cli evaluate                   # metrics for hybrid / semantic / keyword on index/
    python -m src.rag.cli evaluate --index-dir index_minilm
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import EVAL_K, GROUND_TRUTH, INDEX_DIR, JD_DIR, LATENCY_RUNS, METRICS, MODES, RELEVANT_GRADE
from src.file_system.file_tools import read_file
from .job_matcher import JobMatcher


def load_ground_truth(path: Path = GROUND_TRUTH) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text())["judgments"]


def ranked_ids(result: dict) -> list[str]:
    return [Path(m["resume_path"]).stem for m in result["top_matches"]]


def metrics_at_k(ranking: list[str], rel: dict[str, int], k: int = EVAL_K) -> dict[str, float]:
    """Precision@5/10 and Recall@10 count grade>=1 as relevant; nDCG uses graded relevance (2/1/0)."""
    relevant = {r for r, g in rel.items() if g >= RELEVANT_GRADE}
    top = ranking[:k]
    hits = [r in relevant for r in top]
    p5 = sum(hits[:5]) / 5
    p10 = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum((2 ** rel.get(r, 0) - 1) / math.log2(i + 2) for i, r in enumerate(top))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return {"P@5": p5, "P@10": p10, "R@10": recall, "MRR": mrr, "nDCG@10": dcg / idcg if idcg else 0.0}


def evaluate(matcher: JobMatcher, modes=MODES, jd_dir: str | Path = JD_DIR,
             apply_filters: bool = False) -> pd.DataFrame:
    """Per-JD metrics for each retrieval mode. Filters are off by default so we measure ranking quality."""
    gt = load_ground_truth(Path(jd_dir) / "ground_truth.json")
    rows = []
    for jd_file, rel in gt.items():
        text = read_file(Path(jd_dir) / jd_file)
        for mode in modes:
            res = matcher.match(text, k=EVAL_K, mode=mode, apply_filters=apply_filters)
            rows.append({"jd": jd_file.removesuffix(".txt"), "mode": mode,
                         **metrics_at_k(ranked_ids(res), rel)})
    return pd.DataFrame(rows)


def measure_latency(matcher: JobMatcher, runs: int = LATENCY_RUNS, jd_dir: str | Path = JD_DIR) -> pd.DataFrame:
    """Cycle through the JDs `runs` times total, recording per-stage timings (ms)."""
    texts = [read_file(p) for p in sorted(Path(jd_dir).glob("*.txt"))]
    rows = []
    for i in range(runs):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        res = matcher.match(t, k=EVAL_K)
        wall = (time.perf_counter() - t0) * 1000
        rows.append({**res["stats"]["timings_ms"], "wall": wall})
    return pd.DataFrame(rows)


def summarize_latency(df: pd.DataFrame) -> pd.DataFrame:
    return df.agg([lambda s: np.percentile(s, 50), lambda s: np.percentile(s, 95), "mean"]).set_axis(
        ["p50_ms", "p95_ms", "mean_ms"]).T.round(2)


def cli(argv: list[str] | None = None) -> None:
    from .resume_rag import ResumeRAG

    ap = argparse.ArgumentParser(prog="src.rag.cli evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--runs", type=int, default=LATENCY_RUNS)
    args = ap.parse_args(argv)
    m = JobMatcher(ResumeRAG.load(args.index_dir))
    df = evaluate(m)
    print(df.groupby("mode")[METRICS].mean().round(3))
    print(summarize_latency(measure_latency(m, args.runs)))


if __name__ == "__main__":
    cli()
