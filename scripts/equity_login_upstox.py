"""Run this once every trading morning before paper/live trading with Upstox.

Upstox access tokens expire daily (~3:30am IST invalidation), same idea as
Zerodha's daily login (scripts/login.py). Opens the login URL, exchanges
the redirected `code` for an access_token, and writes it into .env.
"""
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

REDIRECT_URI = "https://127.0.0.1/callback"  # must match what's registered in your Upstox app


def main():
    api_key = os.getenv("UPSTOX_API_KEY")
    api_secret = os.getenv("UPSTOX_API_SECRET")

    if not api_key or not api_secret:
        print("Set UPSTOX_API_KEY and UPSTOX_API_SECRET in .env first (see .env.example).")
        sys.exit(1)

    login_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={api_key}&redirect_uri={REDIRECT_URI}"
    )
    print("1. Open this URL in a browser and log in with your Upstox credentials:\n")
    print(f"   {login_url}\n")
    print("2. After login you'll be redirected to a URL like:")
    print(f"   {REDIRECT_URI}/?code=XXXXXX")
    print("3. Paste the FULL redirected URL (or just the code value) below.\n")

    raw = input("Paste redirect URL or code: ").strip()
    match = re.search(r"code=([^&]+)", raw)
    code = match.group(1) if match else raw

    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        data={
            "code": code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    access_token = resp.json()["access_token"]

    if not ENV_PATH.exists():
        ENV_PATH.write_text("")
    set_key(str(ENV_PATH), "UPSTOX_ACCESS_TOKEN", access_token)

    print(f"\nAccess token saved to {ENV_PATH}. Valid until ~3:30am IST tomorrow.")


if __name__ == "__main__":
    main()
