"""Single entry point for getting a broker instance. Everything else in the
codebase should call get_broker(cfg) instead of importing a specific
adapter - that's what makes swapping brokers a config-only change.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT / ".env")


def get_broker(cfg: dict):
    broker_name = cfg["broker"]["name"].lower()

    if broker_name == "zerodha":
        from exchange.brokers.zerodha_broker import ZerodhaBroker
        api_key = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            raise RuntimeError(
                "KITE_API_KEY / KITE_ACCESS_TOKEN not set. Run scripts/login.py "
                "after filling KITE_API_KEY and KITE_API_SECRET in .env.")
        return ZerodhaBroker(api_key, access_token)

    if broker_name == "upstox":
        from exchange.brokers.upstox_broker import UpstoxBroker
        access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        if not access_token:
            raise RuntimeError(
                "UPSTOX_ACCESS_TOKEN not set. Run scripts/login_upstox.py "
                "after filling UPSTOX_API_KEY and UPSTOX_API_SECRET in .env.")
        return UpstoxBroker(access_token)

    if broker_name == "csv":
        from exchange.brokers.csv_broker import CsvBroker
        return CsvBroker(ROOT / "cache" / "kaggle")

    if broker_name == "dhan":
        from exchange.brokers.dhan_broker import DhanBroker
        client_id = os.getenv("DHAN_CLIENT_ID")
        access_token = os.getenv("DHAN_ACCESS_TOKEN")
        if not client_id or not access_token:
            raise RuntimeError(
                "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set in .env. Dhan tokens "
                "are generated from the Dhan web platform under DhanHQ Trading APIs.")
        return DhanBroker(client_id, access_token)

    raise ValueError(f"Unknown broker '{broker_name}'. Use zerodha, upstox, csv, or dhan.")
