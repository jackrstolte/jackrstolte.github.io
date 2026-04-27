import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
import pandas as pd
import warnings
import torch
from transformers import pipeline 

def fetch_public_law_text(congress, billType, billNumber, api_key, base_url):
    '''Fetches the text of a public law given its congress, bill type, and bill number.
    Set base URLS BASE_URL = "https://api.congress.gov/v3" and API_KEY'''
    billType_lower = billType.lower()
    endpoint = f"{base_url}/bill/{congress}/{billType_lower}/{billNumber}/text"
    params = {"api_key": api_key, "format": "json"}

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        text_versions = data.get('textVersions', [])
        for version in text_versions:
            if version.get('type') == 'Public Law':
                formats = version.get('formats', [])
                # Look for XML first
                xml_format = next((f for f in formats if 'xml' in f.get('type', '').lower()), None)
                target_format = xml_format if xml_format else (formats[-1] if formats else None)

                if target_format and 'url' in target_format:
                    text_url = target_format['url']
                    content_url = f"{text_url}?api_key={api_key}" if '?' not in text_url else f"{text_url}&api_key={api_key}"
                    text_response = requests.get(content_url)
                    text_response.raise_for_status()

                    # Use lxml for better XML parsing if xml_format exists
                    parser = 'lxml' if xml_format else 'html.parser'
                    soup = BeautifulSoup(text_response.content, parser)
                    return soup.get_text(separator=' ', strip=True)

        return "Public Law text not found"
    except Exception as e:
        return f"Error: {str(e)}"
    
def sep_billID(df_bills): 
  df_bills[['billType', 'billNumber']] = df_bills['billID'].str.extract(r'([A-Za-z]+)(\d+)')

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