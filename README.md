# Congress Voting Dashboard

An automated pipeline and interactive web dashboard that tracks congressional voting records and computes partisanship scores for every member of Congress across 11 issue categories.

Live site: [jackrstolte.github.io](https://jackrstolte.github.io)

---

## Overview

The project collects roll-call votes on bills that became public law from Congresses 116–119 (2019–present), classifies each vote as partisan or nonpartisan, and produces a per-member partisanship score on a −100 to +100 scale. Scores are broken down by issue area and updated weekly via GitHub Actions.

**Score interpretation**
- **−100** — member always votes with the Democratic majority
- **+100** — member always votes with the Republican majority

---

## Pipeline

The weekly pipeline runs every Sunday at 6 AM UTC via `.github/workflows/pipeline.yml`. `main.py` orchestrates three steps in order:

### 1. `collection.py` — Data collection

- Fetches all public laws for Congresses 116–119 from the Congress.gov API, with caching in `laws_cache.csv` (inactive congresses are never re-fetched).
- For each law, fetches the corresponding Senate roll-call votes from `senate.gov` and House roll-call votes from `clerk.house.gov`.
- New votes are appended to `votes_cache.csv` and `data/congress_data.csv`; already-cached votes are skipped.

### 2. `partisan_checker.py` — Partisanship classification

- Reads `data/congress_data.csv` and keeps only the most recent vote per `(bill_id, chamber, congress)`.
- A vote is **partisan** if the Democratic majority and Republican majority voted differently (Yea vs. Nay); otherwise it is **nonpartisan**.
- Classified votes are moved to `data/partisan_votes.csv` or `data/nonpartisan_votes.csv`.

### 3. `score_calculator.py` — Score computation

- Reads `data/partisan_votes.csv` and looks up each bill's issue category from `classified_bills.csv`.
- For each member vote on a partisan bill:
  - Vote matches Democratic majority → +0 to that issue's total score, +1 to vote count
  - Vote matches Republican majority → +1 to that issue's total score, +1 to vote count
  - Abstentions / "Not Voting" → skipped
- Derives `mean_score = total_score / vote_totals` and `final_score = (mean_score × 200) − 100`.
- Results are written to `data/scores.csv` (~796 members); processed rows move to `data/processed_votes.csv`.

### `classification.py` — Bill categorization 

Uses the `facebook/bart-large-mnli` zero-shot classifier to assign each public law to one of 11 issue categories.

**Issue categories:** Immigration, Healthcare, Taxes/spending/budget, Education, Climate/environment, Nominations, Entitlements (welfare), Military/national security, Technology, Business/employment, Miscellaneous

---

## Data files

| File | Description |
|---|---|
| `votes_cache.csv` | All fetched roll-call votes (~282k records) |
| `laws_cache.csv` | Public laws per congress (~1,039 laws) |
| `classified_bills.csv` | AI-assigned issue categories per bill |
| `data/congress_data.csv` | Votes pending classification |
| `data/partisan_votes.csv` | Partisan votes pending score computation |
| `data/nonpartisan_votes.csv` | Nonpartisan votes (excluded from scoring) |
| `data/processed_votes.csv` | Votes already incorporated into scores |
| `data/scores.csv` | Final per-member partisanship scores |

