"""Rebuild data/job_descriptions/ground_truth.json from the resume categories in data/resumes/_labels.json.

Run this after fetching a new set of resumes (the old ground truth lists the old resume names).
Relevance per JD: 2 = same occupational category, 1 = adjacent category, unlisted = 0.

Usage (from the project root):
    python scripts/build_ground_truth.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from src.config.constants import GROUND_TRUTH, LABELS_FILE, RESUMES_DIR  # noqa: E402

# job description -> {resume category: relevance grade}
JD_CATEGORIES = {
    "it_systems_admin.txt": {"INFORMATION-TECHNOLOGY": 2},
    "staff_accountant.txt": {"ACCOUNTANT": 2, "FINANCE": 1},
    "finance_manager.txt": {"FINANCE": 2, "ACCOUNTANT": 1},
    "hr_generalist.txt": {"HR": 2},
    "sales_associate.txt": {"SALES": 2},
    "graphic_designer.txt": {"DESIGNER": 2},
    "teacher.txt": {"TEACHER": 2},
    "quality_engineer.txt": {"ENGINEERING": 2},
}


def main() -> None:
    labels = json.loads((RESUMES_DIR / LABELS_FILE).read_text())
    judgments = {}
    for jd, grades in JD_CATEGORIES.items():
        judgments[jd] = {Path(fname).stem: grades[info["category"]]
                         for fname, info in sorted(labels.items()) if info["category"] in grades}
    GROUND_TRUTH.write_text(json.dumps({
        "_notes": "Graded relevance per JD: 2 = same occupational category, 1 = adjacent category. "
                  "Unlisted = 0. Generated from resumes/_labels.json by scripts/build_ground_truth.py.",
        "judgments": judgments,
    }, indent=2))
    print(f"Wrote {GROUND_TRUTH} ({sum(len(j) for j in judgments.values())} judgments for {len(judgments)} JDs)")


if __name__ == "__main__":
    main()
