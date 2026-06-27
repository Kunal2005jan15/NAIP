# =============================================================
# NAIP Project - Step 31: Live Policy/News RAG Grounding
# =============================================================
# When a district shows an urgent/watch alert, this module
# searches the LIVE web (Tavily API) for:
#   1. Government drought/flood relief schemes (highest priority)
#   2. MSP / mandi price information (economic context)
#   3. IMD meteorological forecasts (near-term outlook)
# and returns short, grounded, CITED summaries to append to the
# advisory panel - distinct from the static rule-based advisories,
# which remain unchanged and still work without internet/API access.
#
# VALIDATED FINDING (logged during testing, 2026-06-27): this
# module surfaced IMD's official 2026 Southwest Monsoon outlook
# (90% of LPA, 60% chance of deficient rainfall, revised 29 May
# 2026) - a genuine, citable, FORWARD-LOOKING national forecast.
# This complements the system's own rainfall_anomaly_index, which
# is BACKWARD-looking (computed only after rainfall is realized).
# Combining both gives the system earlier warning lead time than
# realized-data-only approaches - a real architectural advantage,
# not just a UI nicety. Worth stating explicitly in the paper.
#
# DESIGN PRINCIPLE: This module must NEVER crash the dashboard if
# the API key is missing or the request fails - it degrades
# gracefully to "live grounding unavailable" rather than breaking
# the core (already-working) prediction system.
# =============================================================

import os
import json

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# ---------------------------------------------------------------
# API KEY HANDLING
# ---------------------------------------------------------------
# Set your key as an environment variable (recommended, never
# hardcode it in committed code):
#   Windows:  set TAVILY_API_KEY=tvly-xxxxxxxx
#   Or create a file data/secrets/tavily_key.txt with just the key

def get_tavily_client():
    api_key = os.environ.get('TAVILY_API_KEY')

    if not api_key:
        key_file = 'data/secrets/tavily_key.txt'
        if os.path.exists(key_file):
            with open(key_file) as f:
                api_key = f.read().strip()

    if not api_key or not TAVILY_AVAILABLE:
        return None

    try:
        return TavilyClient(api_key=api_key)
    except Exception:
        return None


# ---------------------------------------------------------------
# QUERY BUILDERS — one per priority category
# ---------------------------------------------------------------

def build_queries(state, district, crop, alert_type):
    """
    alert_type: 'drought', 'flood', or 'general'
    Returns queries in PRIORITY ORDER: schemes -> market -> weather
    """
    queries = []

    if alert_type == 'drought':
        queries.append(f"{state} drought relief scheme farmers 2026")
    elif alert_type == 'flood':
        queries.append(f"{state} flood relief crop damage compensation 2026")
    else:
        queries.append(f"{state} agriculture relief scheme {crop} 2026")

    queries.append(f"{crop} MSP minimum support price India 2026")
    queries.append(f"IMD {state} monsoon weather forecast 2026")

    return queries


# ---------------------------------------------------------------
# MAIN GROUNDING FUNCTION
# ---------------------------------------------------------------

def get_live_grounding(state, district, crop, alert_type='general', max_results_per_query=2):
    """
    Returns a dict:
      {
        'available': bool,
        'sections': [
          {'category': 'Government Relief Schemes', 'results': [...]},
          {'category': 'Market Prices (MSP)', 'results': [...]},
          {'category': 'Weather Outlook (IMD)', 'results': [...]},
        ]
      }
    Each result: {'title':, 'content':, 'url':}
    Degrades gracefully (available=False) if no API key or on error.
    """
    client = get_tavily_client()
    if client is None:
        return {'available': False, 'reason': 'No Tavily API key configured or tavily-python not installed.', 'sections': []}

    queries = build_queries(state, district, crop, alert_type)
    category_names = ["Government Relief Schemes", "Market Prices (MSP)", "Weather Outlook (IMD)"]

    sections = []
    for query, category in zip(queries, category_names):
        try:
            response = client.search(
                query=query,
                max_results=max_results_per_query,
                search_depth="basic",
            )
            results = [
                {
                    'title': r.get('title', ''),
                    'content': r.get('content', '')[:280],  # keep snippets short for UI
                    'url': r.get('url', ''),
                }
                for r in response.get('results', [])
            ]
            sections.append({'category': category, 'query': query, 'results': results})
        except Exception as e:
            sections.append({'category': category, 'query': query, 'results': [], 'error': str(e)})

    return {'available': True, 'sections': sections}


# ---------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Testing live grounding module...")
    print(f"Tavily library installed: {TAVILY_AVAILABLE}")

    client = get_tavily_client()
    print(f"API key found and client initialized: {client is not None}")

    if client is None:
        print("\nTo enable live grounding:")
        print("  1. pip install tavily-python")
        print("  2. Get a free API key at tavily.com (1000 free credits/month)")
        print("  3. Run: set TAVILY_API_KEY=your-key-here  (Windows, same terminal session)")
        print("     OR save the key to data/secrets/tavily_key.txt")
    else:
        print("\nRunning a live test query for Sirsa, Haryana (drought scenario)...")
        result = get_live_grounding("Haryana", "Sirsa", "Wheat", alert_type='drought')
        print(json.dumps(result, indent=2)[:3000])