"""
score_calculator.py
--------------------
Processes partisan_votes.csv to produce partisanship scores for every
member of Congress across 11 issue categories.

Steps
-----
1. For each unique (congress, bill_id) in partisan_votes.csv, look up the
   AI-assigned issue category from classified_bills.csv. Bills not found in
   classified_bills.csv are assigned "Miscellaneous" and a warning is printed.
2. Row by row, accumulate scores into scores.csv:
     - vote == generic_d_vote  ->  add 0 to issue total_score, +1 to vote_totals
     - vote == generic_r_vote  ->  add 1 to issue total_score, +1 to vote_totals
     - any other vote value    ->  skip (abstentions don't signal ideology)
3. Recompute mean_score and final_score for the affected issue column.
4. Move every processed row from partisan_votes.csv to processed_votes.csv.

Score formula
-------------
  mean_score  = total_score / vote_totals          (range 0..1)
  final_score = (mean_score * 200) - 100           (range -100..+100)
  -100 = always votes with Democrats
  +100 = always votes with Republicans

File layout (relative to this script):
    classified_bills.csv          <- AI classifications (congress, bill_id_clean, predicted_category)
    data/partisan_votes.csv
    data/processed_votes.csv
    data/scores.csv
"""

import os
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT_DIR          = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR          = os.path.join(_ROOT_DIR, "data")

CLASSIFIED_PATH    = os.path.join(_ROOT_DIR, "classified_bills.csv")
PARTISAN_PATH      = os.path.join(_DATA_DIR, "partisan_votes.csv")
PROCESSED_PATH     = os.path.join(_DATA_DIR, "processed_votes.csv")
SCORES_PATH        = os.path.join(_DATA_DIR, "scores.csv")

# ── Issue categories ──────────────────────────────────────────────────────────

ISSUES = [
    "Immigration",
    "Healthcare",
    "Taxes/spending/budget",
    "Education",
    "Climate/environment",
    "Nominations",
    "Entitlements (welfare)",
    "Military/national security",
    "Technology",
    "Business/employment",
    "Miscellaneous",
]

def _col(issue: str, suffix: str) -> str:
    """
    Build a safe column name from an issue label and a suffix.
    e.g. ("Taxes/spending/budget", "total_score") -> "Taxes_spending_budget_total_score"
    """
    safe = (
        issue
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    return f"{safe}_{suffix}"

# Pre-build all score column names in a consistent order.
SCORE_COLS = []
for _issue in ISSUES:
    SCORE_COLS += [
        _col(_issue, "total_score"),
        _col(_issue, "vote_totals"),
        _col(_issue, "mean_score"),
        _col(_issue, "final_score"),
    ]

# Full ordered list of columns for scores.csv
SCORES_COLUMNS = ["member", "bioguide_id", "party", "state"] + SCORE_COLS


# ── Step 1: Assign AI issue labels ────────────────────────────────────────────

def assign_issue_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every unique (congress, bill_id) in df, look up the predicted_category
    from classified_bills.csv.  The join is on:
        partisan_votes.congress  == classified_bills.congress
        partisan_votes.bill_id   == classified_bills.bill_id_clean

    Bills with no match are assigned "Miscellaneous" and logged as warnings.
    Bills that already have an Issue label (from a prior partial run) are
    left unchanged.
    """
    if "Issue" not in df.columns:
        df["Issue"] = None

    # Load AI classifications
    if not os.path.exists(CLASSIFIED_PATH):
        print(f"WARNING: {CLASSIFIED_PATH} not found. All bills will be labelled 'Miscellaneous'.")
        df["Issue"] = df["Issue"].fillna("Miscellaneous")
        return df

    classified = pd.read_csv(CLASSIFIED_PATH)
    classified.columns = [c.lstrip("\ufeff").strip() for c in classified.columns]

    # Build lookup: (congress, bill_id_clean) -> predicted_category
    lookup = (
        classified
        .drop_duplicates(subset=["congress", "bill_id_clean"])
        .set_index(["congress", "bill_id_clean"])["predicted_category"]
        .to_dict()
    )

    # Only fill rows that don't already have a label
    needs_label = df["Issue"].isna()
    unmatched = set()

    def _lookup(row):
        if not needs_label[row.name]:
            return row["Issue"]
        key = (row["congress"], row["bill_id"])
        category = lookup.get(key)
        if category is None:
            unmatched.add(key)
            return "Miscellaneous"
        return category

    df["Issue"] = df.apply(_lookup, axis=1)

    if unmatched:
        print(f"WARNING: {len(unmatched)} (congress, bill_id) pair(s) not found in "
              f"classified_bills.csv — assigned 'Miscellaneous':")
        for key in sorted(unmatched):
            print(f"  congress={key[0]}, bill_id={key[1]}")

    return df


# ── Step 2 & 3: Score accumulation ───────────────────────────────────────────

def _load_scores() -> pd.DataFrame:
    """Load scores.csv if it exists, otherwise return an empty template."""
    if os.path.exists(SCORES_PATH):
        return pd.read_csv(SCORES_PATH)
    return pd.DataFrame(columns=SCORES_COLUMNS)


def _member_key(row: pd.Series) -> str:
    """Unique identifier for a member (bioguide_id if present, else name)."""
    if pd.notna(row.get("bioguide_id")):
        return str(row["bioguide_id"])
    return str(row["member"])


def _ensure_member(scores: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Add a new member row to scores if they don't already exist."""
    key = _member_key(row)
    mask = (
        (scores["bioguide_id"].astype(str) == key) |
        (scores["member"].astype(str) == key)
    )
    if not mask.any():
        new_row = {col: 0 for col in SCORE_COLS}
        new_row["member"]      = row.get("member", "")
        new_row["bioguide_id"] = row.get("bioguide_id", None)
        new_row["party"]       = row.get("party", "")
        new_row["state"]       = row.get("state", "")
        scores = pd.concat([scores, pd.DataFrame([new_row])], ignore_index=True)
    return scores


def _member_mask(scores: pd.DataFrame, row: pd.Series) -> pd.Series:
    """Boolean mask that selects the member's row in scores."""
    key = _member_key(row)
    return (
        (scores["bioguide_id"].astype(str) == key) |
        (scores["member"].astype(str) == key)
    )


def _update_score(scores: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """
    Apply a single partisan vote row to the scores DataFrame.

    - vote == generic_d_vote  ->  +0 to total_score, +1 to vote_totals
    - vote == generic_r_vote  ->  +1 to total_score, +1 to vote_totals
    - anything else           ->  no change (abstention / not voting)
    """
    member_vote = row.get("vote")
    d_vote      = row.get("generic_d_vote")
    r_vote      = row.get("generic_r_vote")
    issue       = row.get("Issue")

    if member_vote not in (d_vote, r_vote):
        return scores   # abstention – skip

    score_increment = 1 if member_vote == r_vote else 0

    col_total  = _col(issue, "total_score")
    col_totals = _col(issue, "vote_totals")
    col_mean   = _col(issue, "mean_score")
    col_final  = _col(issue, "final_score")

    mask = _member_mask(scores, row)

    scores.loc[mask, col_total]  += score_increment
    scores.loc[mask, col_totals] += 1

    # Recompute derived columns for this member / issue
    total  = scores.loc[mask, col_total].values[0]
    totals = scores.loc[mask, col_totals].values[0]

    mean_score  = total / totals if totals > 0 else 0.0
    final_score = (mean_score * 200) - 100

    scores.loc[mask, col_mean]  = round(mean_score,  6)
    scores.loc[mask, col_final] = round(final_score, 4)

    return scores


# ── Main function ─────────────────────────────────────────────────────────────

def calculate_scores() -> None:
    """
    Full pipeline:
      1. Assign AI Issue labels to any unlabeled bills in partisan_votes.csv.
      2. Process every vote row, updating scores.csv.
      3. Move all processed rows to processed_votes.csv.
      4. Clear processed rows from partisan_votes.csv.
    """

    # ── Load ─────────────────────────────────────────────────────────────────
    if not os.path.exists(PARTISAN_PATH):
        print("partisan_votes.csv not found – nothing to process.")
        return

    partisan_df = pd.read_csv(PARTISAN_PATH)
    partisan_df.columns = [c.lstrip("\ufeff").strip() for c in partisan_df.columns]

    if partisan_df.empty:
        print("partisan_votes.csv is empty – nothing to process.")
        return

    # ── Step 1: Issue labels from classified_bills.csv ───────────────────────
    partisan_df = assign_issue_labels(partisan_df)
    # Persist labels immediately so a crash mid-run doesn't lose them.
    partisan_df.to_csv(PARTISAN_PATH, index=False)

    # ── Step 2 & 3: Score accumulation ───────────────────────────────────────
    scores = _load_scores()
    processed_rows = []

    for _, vote_row in partisan_df.iterrows():
        scores = _ensure_member(scores, vote_row)
        scores = _update_score(scores, vote_row)
        processed_rows.append(vote_row)

    # ── Save scores.csv ───────────────────────────────────────────────────────
    scores.to_csv(SCORES_PATH, index=False)

    # ── Step 4: Move processed rows to processed_votes.csv ───────────────────
    if processed_rows:
        processed_df = pd.DataFrame(processed_rows)
        write_header = not os.path.exists(PROCESSED_PATH)
        processed_df.to_csv(PROCESSED_PATH, mode="a", index=False, header=write_header)

    # Clear partisan_votes.csv (keep header, remove all rows)
    pd.DataFrame(columns=partisan_df.columns).to_csv(PARTISAN_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("Done.")
    print(f"  Votes processed                   : {len(processed_rows)}")
    print(f"  Members in scores.csv             : {len(scores)}")
    print(f"  Rows moved to processed_votes.csv : {len(processed_rows)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    calculate_scores()
