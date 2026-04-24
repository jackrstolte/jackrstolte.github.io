import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
import warnings

def fetch_public_law_text(congress, billType, billNumber, api_key, base_url):
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