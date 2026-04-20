import requests
import pandas as pd
import xml.etree.ElementTree as ET
import re
import time
import os

# =========================
# CONFIG
# =========================

API_KEY = "pMAZYBmweGSn33zhfhmWnTxzqYeHHoAXqH3RnKY0"
CONGRESS = 119

CONGRESS_API = "https://api.congress.gov/v3/bill"
SENATE_BASE = "https://www.senate.gov/legislative/LIS/roll_call_votes"

OUTPUT_FILE = "congress_pipeline.csv"

HEADERS = {
    "X-API-Key": API_KEY,
    "User-Agent": "Mozilla/5.0"
}


# =========================
# NORMALIZATION
# =========================

def normalize_bill_id(bill):
    if not isinstance(bill, str):
        return None
    return re.sub(r"[.\s]", "", bill.lower())


# =========================
# 1. GET LAWS
# =========================

def get_law_bills(congress):
    all_bills = []
    offset = 0
    limit = 250

    while True:
        url = (
            f"{CONGRESS_API}/{congress}"
            f"?offset={offset}&limit={limit}&format=json"
        )

        r = requests.get(url, headers=HEADERS)

        if r.status_code != 200:
            print("Congress API error:", r.status_code)
            print(r.text[:300])
            break

        try:
            data = r.json()
        except ValueError:
            print("Failed to parse JSON")
            print(r.text[:300])
            break

        bills = data.get("bills", [])

        # Stop when no more results
        if not bills:
            break

        for bill in bills:
            latest_action_obj = bill.get("latestAction") or {}
            latest_action_text = latest_action_obj.get("text", "")
            latest_action = latest_action_text.lower()

            if any(x in latest_action for x in ["became law", "public law"]):
                bill_type = bill.get("type", "")
                bill_number = bill.get("number", "")
                bill_id_clean = f"{bill_type}{bill_number}"

                all_bills.append({
                    "bill_id_clean": bill_id_clean,
                    "congress": bill.get("congress"),
                    "title": bill.get("title"),
                    "origin_chamber": bill.get("originChamber"),
                    "latest_action": latest_action_text,
                    "action_date": latest_action_obj.get("actionDate"),
                })

        offset += limit
        time.sleep(0.2)

    return pd.DataFrame(all_bills)


# =========================
# 2. SENATE VOTES 
# =========================

def get_senate_votes(congress):
    votes = []

    for session in [1, 2]:
        print(f"Senate session {session}")

        for i in range(1, 2000):
            vote_num = str(i).zfill(5)

            url = f"{SENATE_BASE}/vote{congress}{session}/vote_{congress}_{session}_{vote_num}.xml"

            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

            if r.status_code == 404:
                continue
            if r.status_code != 200:
                break

            xml_text = r.text

            # Fix common XML issues
            xml_text = xml_text.replace("&", "&amp;")

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                print("Bad XML at:", url)
                print(xml_text[:500])
                continue

            bill = root.findtext(".//document_name") or ""
            bill_clean = normalize_bill_id(bill)

            vote_number = root.findtext(".//vote_number")

            for m in root.findall(".//member"):
                first = m.findtext("first_name") or ""
                last = m.findtext("last_name") or ""
                full_name = f"{first} {last}".strip()

                votes.append({
                    "bill_id_clean": bill_clean,
                    "vote_number": vote_number,
                    "member_id": m.findtext("lis_member_id"),
                    "name": full_name,
                    "party": m.findtext("party"),
                    "state": m.findtext("state"),
                    "vote": m.findtext("vote_cast"),
                    "chamber": "senate"
                })

            time.sleep(0.1)

    return pd.DataFrame(votes)


# =========================
# 3. HOUSE VOTES
# =========================

def get_house_votes():
    all_votes = []

    for i in range(1, 1000):
        vote_num = str(i).zfill(3)
        url = f"https://clerk.house.gov/evs/2025/roll{vote_num}.xml"

        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        if r.status_code != 200:
            break

        root = ET.fromstring(r.content)

        metadata = root.find(".//vote-metadata")
        if metadata is None:
            continue

        legis_num = metadata.findtext("legis-num")
        bill_clean = normalize_bill_id(legis_num)

        vote_desc = metadata.findtext("vote-desc")
        vote_question = metadata.findtext("vote-question")

        for v in root.findall(".//recorded-vote"):
            leg = v.find("legislator")
            choice = v.find("vote")

            all_votes.append({
                "roll_call": i,
                "bill_id_clean": bill_clean,
                "bill_raw": legis_num,
                "member_id": leg.attrib.get("name-id") if leg is not None else None,
                "name": leg.text if leg is not None else None,
                "party": leg.attrib.get("party") if leg is not None else None,
                "state": leg.attrib.get("state") if leg is not None else None,
                "vote": choice.text if choice is not None else None,
                "description": vote_desc,
                "question": vote_question,
                "chamber": "house"
            })

        time.sleep(0.2)

    return pd.DataFrame(all_votes)


# =========================
# 4. LOAD EXISTING DATA
# =========================

def load_existing():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE)
    return pd.DataFrame()


# =========================
# 5. PIPELINE
# =========================

def run_pipeline():
    print("Loading existing dataset...")
    old_data = load_existing()

    print("Fetching law bills...")
    laws = get_law_bills(CONGRESS)

    law_set = set(laws["bill_id_clean"]) if not laws.empty else set()

    print("Fetching Senate votes...")
    senate = get_senate_votes(CONGRESS)

    print("Fetching House votes...")
    house = get_house_votes()

    # =========================
    # FILTER AFTER NORMALIZATION (IMPORTANT FIX)
    # =========================

    if not senate.empty:
        senate = senate[senate["bill_id_clean"].isin(law_set)]

    if not house.empty:
        house = house[house["bill_id_clean"].isin(law_set)]

    # =========================
    # MERGE ALL
    # =========================

    new_data = pd.concat([senate, house], ignore_index=True)

    # enrich with law metadata
    if not laws.empty:
        new_data = new_data.merge(laws, on="bill_id_clean", how="left")

    # =========================
    # DEDUPE
    # =========================

    combined = pd.concat([old_data, new_data], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["bill_id_clean", "member_id", "vote_number"],
        keep="last"
    )

    print(f"Saving {len(combined)} rows...")
    combined.to_csv(OUTPUT_FILE, index=False)

    print("Done.")


if __name__ == "__main__":
    run_pipeline()