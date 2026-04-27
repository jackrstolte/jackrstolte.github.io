import requests
import pandas as pd
import xml.etree.ElementTree as ET
import re
import time
import os
from dotenv import load_dotenv
from requests.exceptions import ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError

# =========================
# CONFIG
# =========================
load_dotenv()
API_KEY = os.getenv("CONGRESS_API_KEY")
CONGRESSES = [116, 117, 118, 119]
OUTPUT_FILE = "data/congress_data.csv"
LAWS_CACHE = "laws_cache.csv"
VOTES_CACHE = "votes_cache.csv" 
HEADERS = {"X-API-Key": API_KEY, "User-Agent": "Mozilla/5.0 (PipelineBot/1.0)"}

if not API_KEY:
    raise ValueError("Missing CONGRESS_API_KEY")

CONGRESS_MAP = {
    116: {"years": [2019, 2020], "sessions": [1, 2], "active": False},
    117: {"years": [2021, 2022], "sessions": [1, 2], "active": False},
    118: {"years": [2023, 2024], "sessions": [1, 2], "active": False},
    119: {"years": [2025, 2026], "sessions": [1, 2], "active": True}
}

# =========================
# UTILITIES
# =========================

def normalize_bill_id(bill_str):
    if not bill_str: return None
    return re.sub(r"[.\s]", "", bill_str).lower()

def get_xml_root(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return ET.fromstring(r.content) if r.status_code == 200 else None
    except: return None

# =========================
# 1. LAW FETCHING (WITH CACHE)
# =========================

def get_public_laws(congress):
    if os.path.exists(LAWS_CACHE):
        cache_df = pd.read_csv(LAWS_CACHE)
        if congress in cache_df['congress'].values and not CONGRESS_MAP[congress]['active']:
            print(f"Loading Congress {congress} laws from cache...")
            return set(cache_df[cache_df['congress'] == congress]['bill_id_clean'])

    laws_found = []
    offset = 0
    limit = 250
    print(f"Scanning API for Congress {congress} laws...")
    
    while True:
        url = f"https://api.congress.gov/v3/bill/{congress}?format=json&offset={offset}&limit={limit}"
        data = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except (ChunkedEncodingError, ProtocolError, RequestException):
                time.sleep(5)
        
        if not data: break
        for bill in data.get("bills", []):
            action_text = (bill.get("latestAction") or {}).get("text", "").lower()
            if "became law" in action_text or "public law" in action_text:
                laws_found.append({
                    "congress": congress,
                    "bill_id_clean": normalize_bill_id(f"{bill.get('type')}{bill.get('number')}"),
                })
        
        total = data.get("pagination", {}).get("count", 0)
        offset += limit
        if offset >= total: break
        time.sleep(0.2)

    new_laws_df = pd.DataFrame(laws_found)
    if os.path.exists(LAWS_CACHE):
        old_cache = pd.read_csv(LAWS_CACHE)
        new_laws_df = pd.concat([old_cache, new_laws_df]).drop_duplicates(subset=['bill_id_clean'])
    new_laws_df.to_csv(LAWS_CACHE, index=False)
    return set(new_laws_df[new_laws_df['congress'] == congress]['bill_id_clean'])

# =========================
# 2. VOTE FETCHING (WITH CACHE)
# =========================

def fetch_senate_votes(congress, session, law_set, existing_vote_ids):
    votes = []
    menu_url = f"https://www.senate.gov/legislative/LIS/roll_call_lists/vote_menu_{congress}_{session}.xml"
    root = get_xml_root(menu_url)
    if root == None: return []
    
    nums = [v.text for v in root.findall(".//vote_number")]
    for num in nums:
        vote_id = f"S-{congress}-{session}-{num}"
        
        # SKIP if already in cache
        if vote_id in existing_vote_ids:
            continue
            
        v_url = f"https://www.senate.gov/legislative/LIS/roll_call_votes/vote{congress}{session}/vote_{congress}_{session}_{num.zfill(5)}.xml"
        v_root = get_xml_root(v_url)
        if v_root == None: continue
        
        bill_id = normalize_bill_id(v_root.findtext(".//document_name"))
        if bill_id in law_set:
            print(f"Adding new Senate Vote: {vote_id} for {bill_id}")
            for m in v_root.findall(".//member"):
                votes.append({
                    "congress": congress, "chamber": "Senate", "vote_id": vote_id,
                    "bill_id": bill_id, "member": f"{m.findtext('first_name')} {m.findtext('last_name')}", "party": m.findtext("party"),      # ✅ added
                    "state": m.findtext("state"), 
                    "vote": m.findtext("vote_cast")
                })
        time.sleep(0.1)
    return votes

def fetch_house_votes(congress, year, law_set, existing_vote_ids):
   
    votes = []
    print(f"--- Scanning House Votes for {year} ---")

    # We loop through possible roll call numbers
    for i in range(1, 1000):
        vote_id = f"H-{year}-{i}"
        
        # 1. SKIP if already in cache (Saves massive time)
        if vote_id in existing_vote_ids:
            continue
            
        vote_num_str = str(i).zfill(3)
        url = f"https://clerk.house.gov/evs/{year}/roll{vote_num_str}.xml"

        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code != 200:
                break

            root = ET.fromstring(r.content)
            
            # 2. Extract Metadata using the paths from your snippet
            metadata = root.find(".//vote-metadata")
            if metadata is None: continue
            
            legis_num = metadata.findtext("legis-num")
            
            # Normalize for matching
            bill_clean = normalize_bill_id(legis_num)

            # 3. ONLY process if it's in our Law Set
            if bill_clean and bill_clean in law_set:
                print(f"Adding new House Vote: {vote_id} for {bill_clean}")
                
                vote_desc = metadata.findtext("vote-desc")
                vote_question = metadata.findtext("vote-question")

                for v in root.findall(".//recorded-vote"):
                    leg = v.find("legislator")
                    choice = v.find("vote")

                    if leg is not None:
                        votes.append({
                            "congress": congress,
                            "chamber": "House",
                            "vote_id": vote_id,
                            "bill_id": bill_clean,
                            "member": leg.text or "Unknown",
                            "bioguide_id": leg.attrib.get("name-id"),
                            "party": leg.attrib.get("party"),
                            "state": leg.attrib.get("state"),
                            "vote": choice.text if choice is not None else "Unknown",
                            "description": vote_desc,
                            "question": vote_question
                        })
            time.sleep(0.1)

        except Exception as e:
            print(f"Error on House Vote {i}: {e}")
            continue

    return votes

# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    existing_vote_ids = set()
    if os.path.exists(VOTES_CACHE):
        existing_df = pd.read_csv(VOTES_CACHE)
        existing_vote_ids = set(existing_df['vote_id'].unique())
        print(f"Cache loaded: {len(existing_vote_ids)} votes already stored.")

    all_new_votes = []
    
    for congress in CONGRESSES:
        law_set = get_public_laws(congress)
        
        # Senate
        for sess in CONGRESS_MAP[congress]["sessions"]:
            all_new_votes.extend(fetch_senate_votes(congress, sess, law_set, existing_vote_ids))
        
        # House
        for yr in CONGRESS_MAP[congress]["years"]:
            all_new_votes.extend(fetch_house_votes(congress, yr, law_set, existing_vote_ids))

    # 2. Update the Votes Cache
    if all_new_votes:
        new_votes_df = pd.DataFrame(all_new_votes)
        if os.path.exists(VOTES_CACHE):
            final_votes_df = pd.concat([pd.read_csv(VOTES_CACHE), new_votes_df])
        else:
            final_votes_df = new_votes_df
            
        final_votes_df.drop_duplicates(subset=["vote_id", "member"], inplace=True)
        final_votes_df.to_csv(VOTES_CACHE, index=False)
        # Final output for your site
        final_votes_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Pipeline complete. Total records: {len(final_votes_df)}")
    else:
        print("No new votes found to add.")

if __name__ == "__main__":
    run_pipeline()