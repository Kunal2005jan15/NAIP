# =============================================================
# NAIP - Tavily API Key Setup (run once)
# =============================================================
# Gets your Tavily key from: https://app.tavily.com/home
# Free tier is fine for this project (1000 searches/month).
# Validates the key works, then saves it to
# data/secrets/tavily_key.txt (which is gitignored - never
# committed to your repo).
# =============================================================

import os
import sys

print("NAIP - Tavily Live Context Setup")
print("=" * 40)
print("Get your free API key at: https://app.tavily.com/home")
print()

try:
    from tavily import TavilyClient
except ImportError:
    print("Installing tavily-python...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tavily-python',
                    '--break-system-packages', '-q'], check=True)
    from tavily import TavilyClient

api_key = input("Paste your Tavily API key here (starts with tvly-): ").strip()

if not api_key.startswith('tvly-'):
    print("That doesn't look like a Tavily key (should start with tvly-).")
    sys.exit(1)

print("\nValidating key with a test search...")
try:
    client = TavilyClient(api_key=api_key)
    result = client.search("India IMD monsoon forecast 2026", max_results=1)
    if result.get('results'):
        print(f"  Key works. Test result: {result['results'][0]['title'][:60]}...")
    else:
        print("  Key accepted but returned no results - may be fine.")
except Exception as e:
    print(f"  Key validation failed: {e}")
    print("  Check that the key is correct and your Tavily account is active.")
    sys.exit(1)

os.makedirs('data/secrets', exist_ok=True)
key_path = 'data/secrets/tavily_key.txt'
with open(key_path, 'w') as f:
    f.write(api_key)

print(f"\nKey saved to {key_path} (gitignored - won't be committed).")
print("The Live Context section in the dashboard will now work automatically.")
print("You can also set TAVILY_API_KEY as a Windows environment variable")
print("for a more permanent setup (recommended for production).")