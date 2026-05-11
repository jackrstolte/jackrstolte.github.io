"""
main.py
-------
Weekly pipeline entry point. Runs in order:
  1. collection.py      – fetch new votes from congress.gov / senate.gov / clerk.house.gov
  2. partisan_checker.py – classify votes as partisan or nonpartisan
  3. score_calculator.py – compute per-member partisanship scores
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _run_step(name: str, fn):
    log.info("=== Step: %s ===", name)
    start = time.time()
    fn()
    elapsed = time.time() - start
    log.info("=== %s complete (%.1fs) ===", name, elapsed)


def main():
    from collection import run_pipeline
    from partisan_checker import process_votes
    from score_calculator import calculate_scores

    steps = [
        ("collection",       run_pipeline),
        ("partisan_checker", process_votes),
        ("score_calculator", calculate_scores),
    ]

    for name, fn in steps:
        try:
            _run_step(name, fn)
        except Exception as exc:
            log.error("Pipeline failed at step '%s': %s", name, exc, exc_info=True)
            sys.exit(1)

    log.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
