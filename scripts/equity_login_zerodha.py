"""Run this once every trading morning before paper/live trading.

Kite Connect access tokens expire daily (~6am IST invalidation). This script
opens the login URL, asks you to paste back the redirected request_token,
exchanges it for an access_token, and writes it into .env automatically.
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)


def main():
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        print("Set KITE_API_KEY and KITE_API_SECRET in .env first (see .env.example).")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print("1. Open this URL in a browser and log in with your Zerodha credentials:\n")
    print(f"   {login_url}\n")
    print("2. After login you'll be redirected to a URL like:")
    print("   https://your-redirect-url/?request_token=XXXXXX&action=login&status=success")
    print("3. Paste the FULL redirected URL (or just the request_token value) below.\n")

    raw = input("Paste redirect URL or request_token: ").strip()
    match = re.search(r"request_token=([^&]+)", raw)
    request_token = match.group(1) if match else raw

    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)

    print(f"\nAccess token saved to {ENV_PATH}. Valid until ~6am IST tomorrow.")


if __name__ == "__main__":
    main()
