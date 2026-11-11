"""
drop_new_resume.py - simulate a new applicant for the watch_directory demo

Copies a sample resume from data/new_resumes/ into data/resumes/ (after an optional delay),
so the agent's "Check for new resumes" sees it arrive.

Run:  python scripts/drop_new_resume.py                  # copy every sample resume now
      python scripts/drop_new_resume.py --delay 5        # wait 5 seconds first
      python scripts/drop_new_resume.py --cleanup        # remove the copied samples again
"""

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "new_resumes"
RESUMES = ROOT / "data" / "resumes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0, help="seconds to wait before copying")
    parser.add_argument("--cleanup", action="store_true", help="remove the sample resumes from data/resumes")
    args = parser.parse_args()

    samples = sorted(p for p in SAMPLES.iterdir() if p.is_file() and not p.name.startswith("."))
    if args.cleanup:
        for sample in samples:
            (RESUMES / sample.name).unlink(missing_ok=True)
            print(f"Removed data/resumes/{sample.name}")
        return

    if args.delay:
        print(f"Waiting {args.delay} s...")
        time.sleep(args.delay)
    for sample in samples:
        shutil.copy(sample, RESUMES / sample.name)
        print(f"New resume arrived: data/resumes/{sample.name}")


if __name__ == "__main__":
    main()
