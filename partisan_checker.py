"""
partisan_checker.py
--------------------
Processes congress_data.csv by:
  1. Keeping only the most recent vote per (bill_id, chamber) pair.
  2. Classifying each remaining vote group as partisan or nonpartisan
     based on whether the Democrat majority and Republican majority
     voted differently.
  3. Moving classified rows out of congress_data.csv and into
     partisan_votes.csv or nonpartisan_votes.csv.

File layout (relative to this script):
    data/congress_data.csv
    data/partisan_votes.csv
    data/nonpartisan_votes.csv
"""

import os
import pandas as pd

# ── Path helpers ─────────────────────────────────────────────────────────────

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "data")

CONGRESS_DATA_PATH   = os.path.join(_DATA_DIR, "congress_data.csv")
PARTISAN_PATH        = os.path.join(_DATA_DIR, "partisan_votes.csv")
NONPARTISAN_PATH     = os.path.join(_DATA_DIR, "nonpartisan_votes.csv")


# ── Vote-ID ordering ─────────────────────────────────────────────────────────

def _vote_id_sort_key(vote_id: str) -> int:
    """
    Return the trailing numeric portion of a vote_id as an integer so that
    vote_ids can be compared by recency.

    Examples
    --------
    "S-118-1-00343" -> 343
    "H-118-1-00021" ->  21
    """
    try:
        return int(vote_id.split("-")[-1])
    except (ValueError, AttributeError):
        return -1


# ── Partisanship logic ───────────────────────────────────────────────────────

def _majority_vote(series: pd.Series) -> str | None:
    """
    Given a Series of "Yea" / "Nay" values, return whichever has a strict
    majority. Returns None on a tie (or if the series is empty).
    """
    counts = series.value_counts()
    yea = counts.get("Yea", 0)
    nay = counts.get("Nay", 0)
    if yea > nay:
        return "Yea"
    if nay > yea:
        return "Nay"
    return None   # tie


def _is_partisan(group: pd.DataFrame) -> bool:
    """
    Decide whether a single (bill_id, chamber) vote group is partisan.

    Rules
    -----
    - Only "D" and "R" members are considered; others (e.g. Independents)
      are excluded from the majority calculation.
    - Only "Yea" and "Nay" votes count; "Not Voting", "Present", etc. are
      excluded.
    - If either party has a tied majority (or no eligible voters), the group
      is treated as nonpartisan.
    - Partisan = Democrat majority ≠ Republican majority.
    """
    eligible = group[group["vote"].isin(["Yea", "Nay"])]

    dem_votes = eligible.loc[eligible["party"] == "D", "vote"]
    rep_votes = eligible.loc[eligible["party"] == "R", "vote"]

    dem_majority = _majority_vote(dem_votes)
    rep_majority = _majority_vote(rep_votes)

    # Either party tied (or no data) → nonpartisan
    if dem_majority is None or rep_majority is None:
        return False

    return dem_majority != rep_majority


# ── Main function ─────────────────────────────────────────────────────────────

def process_votes() -> None:
    """
    Read congress_data.csv, filter to the most recent vote per
    (bill_id, chamber), classify each group as partisan or nonpartisan,
    append the rows to the appropriate output CSV, and rewrite
    congress_data.csv with the classified rows removed.
    """

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CONGRESS_DATA_PATH)

    # Strip any accidental BOM from the first column header
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

    # ── 2. Keep only the most recent vote per (bill_id, chamber) ─────────────
    #
    # Add a numeric sort key so we can pick the max vote_id per group.
    df["_vote_num"] = df["vote_id"].apply(_vote_id_sort_key)

    # For each (bill_id, chamber), find the maximum numeric vote_id.
    latest = (
        df.groupby(["bill_id", "chamber"])["_vote_num"]
        .transform("max")
    )

    # Keep only rows whose vote_num matches the latest for their group.
    df_latest = df[df["_vote_num"] == latest].copy()

    # Rows that belong to older vote_ids — these get removed from the main
    # CSV but are NOT written to partisan/nonpartisan (they're simply dropped).
    df_old = df[df["_vote_num"] != latest].copy()

    # Drop the helper column before writing anything.
    df_latest = df_latest.drop(columns=["_vote_num"])

    # ── 3. Classify each (bill_id, chamber) group ────────────────────────────
    partisan_rows    = []
    nonpartisan_rows = []

    for (bill_id, chamber), group in df_latest.groupby(["bill_id", "chamber"]):
        if _is_partisan(group):
            partisan_rows.append(group)
        else:
            nonpartisan_rows.append(group)

    partisan_df    = pd.concat(partisan_rows)    if partisan_rows    else pd.DataFrame(columns=df_latest.columns)
    nonpartisan_df = pd.concat(nonpartisan_rows) if nonpartisan_rows else pd.DataFrame(columns=df_latest.columns)

    # ── 4. Append to output CSVs ─────────────────────────────────────────────
    #
    # Append (write header only if the file does not yet exist).
    for path, data in [(PARTISAN_PATH, partisan_df), (NONPARTISAN_PATH, nonpartisan_df)]:
        if data.empty:
            continue
        write_header = not os.path.exists(path)
        data.to_csv(path, mode="a", index=False, header=write_header)

    # ── 5. Rewrite congress_data.csv with classified rows removed ────────────
    #
    # Build a set of (bill_id, chamber, vote_id) tuples that were classified
    # so we can filter them out.
    classified_keys = set(
        zip(
            pd.concat([partisan_df, nonpartisan_df])["bill_id"],
            pd.concat([partisan_df, nonpartisan_df])["chamber"],
            pd.concat([partisan_df, nonpartisan_df])["vote_id"],
        )
    ) if not (partisan_df.empty and nonpartisan_df.empty) else set()

    # Also drop old (superseded) vote rows — they no longer belong in the CSV.
    old_keys = set(
        zip(df_old["bill_id"], df_old["chamber"], df_old["vote_id"])
    )

    all_removed_keys = classified_keys | old_keys

    # Filter the original dataframe.
    df_original = df.drop(columns=["_vote_num"])
    mask_remove = df_original.apply(
        lambda row: (row["bill_id"], row["chamber"], row["vote_id"]) in all_removed_keys,
        axis=1,
    )
    df_remaining = df_original[~mask_remove]

    df_remaining.to_csv(CONGRESS_DATA_PATH, index=False)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    n_partisan    = len(partisan_df["vote_id"].unique())    if not partisan_df.empty    else 0
    n_nonpartisan = len(nonpartisan_df["vote_id"].unique()) if not nonpartisan_df.empty else 0

    print(f"Done.")
    print(f"  Partisan vote groups moved    : {n_partisan}")
    print(f"  Nonpartisan vote groups moved : {n_nonpartisan}")
    print(f"  Rows remaining in congress_data.csv: {len(df_remaining)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_votes()
