"""Download a stratified sample of resume PDFs from the public HuggingFace mirror
of the Kaggle "Resume Dataset" (snehaanbhawal/resume-dataset, CC0).

The source resumes are anonymized, so each one gets a deterministic synthetic
name. Files are saved as data/resumes/<first_last>.pdf and the source category is
recorded in data/resumes/_labels.json (used only for evaluation, never for matching).

Usage (from the project root):
    python scripts/fetch_resumes.py --per-category 5
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from src.config.constants import (  # noqa: E402
    FETCH_CATEGORIES, FETCH_PER_CATEGORY, FETCH_SEED, FIRST_NAMES, HF_FILE_URL, HF_TREE_URL, LABELS_FILE,
    LAST_NAMES, RESUMES_DIR,
)
from src.file_system.file_tools import rel  # noqa: E402


def list_pdfs(category: str) -> list[str]:
    resp = requests.get(HF_TREE_URL.format(cat=category), timeout=30)
    resp.raise_for_status()
    return sorted(e["path"] for e in resp.json() if e["path"].endswith(".pdf"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-category", type=int, default=FETCH_PER_CATEGORY)
    ap.add_argument("--out", default=str(RESUMES_DIR))
    ap.add_argument("--seed", type=int, default=FETCH_SEED)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # every first+last combination (2500 names), so 100+ resumes never share a name and overwrite each other
    names = [f"{f} {l}" for f, l in itertools.product(FIRST_NAMES, LAST_NAMES)]
    rng.shuffle(names)

    labels = {}
    i = 0
    for cat in FETCH_CATEGORIES:
        picks = rng.sample(list_pdfs(cat), args.per_category)
        for src in picks:
            name = names[i % len(names)]
            i += 1
            fname = name.lower().replace(" ", "_") + ".pdf"
            data = requests.get(HF_FILE_URL.format(path=src), timeout=60).content
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"Not a PDF: {src}")
            (out / fname).write_bytes(data)
            labels[fname] = {"name": name, "category": cat, "source": src}
            print(f"{cat:24s} {src.split('/')[-1]:14s} -> {fname}")

    (out / LABELS_FILE).write_text(json.dumps(labels, indent=2))
    print(f"\nSaved {len(labels)} resumes to {rel(out)}/")


if __name__ == "__main__":
    main()
