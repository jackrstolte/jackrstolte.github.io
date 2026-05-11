import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
import pandas as pd
import torch
from transformers import pipeline 
import os

CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

def fetch_public_law_text(congress, billType, billNumber, api_key, base_url):
    """
    Fetches public law text via two strategies:
    1. Primary: govinfo.gov API using PLAW package (avoids congress.gov 403s)
    2. Fallback: congress.gov API text endpoint
    
    api_key: your api.data.gov key (works for both congress.gov and govinfo.gov)
    """
    billType_lower = billType.lower()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        endpoint = f"{base_url}/bill/{congress}/{billType_lower}/{billNumber}"
        response = requests.get(endpoint, params={"api_key": api_key, "format": "json"})
        response.raise_for_status()
        data = response.json()

        # Extract public law number e.g. "118-3" -> publ3
        laws = data.get('bill', {}).get('laws', [])
        public_law = next((l for l in laws if l.get('type') == 'Public Law'), None)

        if public_law:
            law_number = public_law.get('number', '')  # e.g. "118-273"
            law_num_only = law_number.split('-')[-1]    # e.g. "273"
            package_id = f"PLAW-{congress}publ{law_num_only}"

            # Fetch HTM text directly from govinfo
            htm_url = f"https://api.govinfo.gov/packages/{package_id}/htm"
            gov_response = requests.get(
                htm_url,
                params={"api_key": api_key},
                headers=headers
            )

            if gov_response.status_code == 200:
                soup = BeautifulSoup(gov_response.content, 'html.parser')
                return soup.get_text(separator=' ', strip=True)

    except Exception as e:
        print(f"govinfo strategy failed for {billType}{billNumber}: {str(e)}")
    try:
        endpoint = f"{base_url}/bill/{congress}/{billType_lower}/{billNumber}/text"
        response = requests.get(endpoint, params={"api_key": api_key, "format": "json"})
        response.raise_for_status()
        data = response.json()

        text_versions = data.get('textVersions', [])
        version_priority = ['public law', 'enrolled bill']
        selected_version = None

        for priority in version_priority:
            selected_version = next(
                (v for v in text_versions if v.get('type', '').lower() == priority), None
            )
            if selected_version:
                break

        if not selected_version and text_versions:
            selected_version = text_versions[0]

        if selected_version:
            formats = selected_version.get('formats', [])
            # Try Formatted Text first, skip uslm xml
            target = next((f for f in formats if f.get('type') == 'Formatted Text'), None)
            if not target:
                target = next((f for f in formats 
                               if 'xml' in f.get('type','').lower() 
                               and 'uslm' not in f.get('url','').lower()), None)
            if not target and formats:
                target = formats[0]

            if target and 'url' in target:
                text_response = requests.get(target['url'], headers=headers)
                text_response.raise_for_status()
                soup = BeautifulSoup(text_response.content, 'html.parser')
                return soup.get_text(separator=' ', strip=True)

        return "Public Law text not found"
    except Exception as e:
        return f"Error: {str(e)}"

    
def sep_billID(df_bills): 
  df_bills[['billType', 'billNumber']] = df_bills['bill_id_clean'].str.extract(r'([A-Za-z]+)(\d+)')

def classify_bills(df_bills):
    categories = [
    "Immigration", "Healthcare", "Taxes/spending/budget", "Education",
    "Climate/environment", "Nominations", "Entitlements (welfare)",
    "Military/national security", "Technology", "Business/employment", "Miscellaneous"]
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device
    )

    results = classifier(df_bills['full_text'].tolist(), candidate_labels=categories, batch_size=len(df_bills))
    df_bills['predicted_category'] = [res['labels'][0] for res in results]
    df_bills['confidence_score'] = [res['scores'][0] for res in results]

def pipe():
    df_bills = pd.read_csv("laws_cache.csv")
    sep_billID(df_bills)
    df_bills['full_text'] = df_bills.apply(lambda row: fetch_public_law_text(row['congress'], row['billType'], row['billNumber'], CONGRESS_API_KEY, "https://api.congress.gov/v3"), axis=1)
    classify_bills(df_bills)
    df_bills=df_bills[['congress', 'bill_id_clean', 'predicted_category']]
    df_bills.to_csv("classified_bills.csv", index=False)

if __name__ == "__main__":
    pipe()
