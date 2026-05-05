import json, sys
from pathlib import Path
from datetime import datetime, timedelta

LOG_DIR = Path("/var/www/html/ML/kat/logs")
trades_path = LOG_DIR / "recent_sim_trades.json"
INITIAL_WALLET_USD = 200
POSITION_SIZE_PCT = 0.05

def test():
    recent_trades = []
    if trades_path.exists():
        try:
            with open(trades_path, "r") as f:
                raw_trades = json.load(f)
            curr_w = INITIAL_WALLET_USD
            comp_trades = []
            for t in reversed(raw_trades):
                try:
                    utc_t = datetime.fromisoformat(t["timestamp"])
                    ist_t = utc_t + timedelta(hours=5, minutes=30)
                    t["timestamp"] = ist_t.strftime("%d %b %Y %I:%M %p")
                except Exception as e:
                    print(f"Time error: {e}")

                pos_s = curr_w * POSITION_SIZE_PCT
                t["value_usd"] = pos_s
                t["pnl_usd"] = pos_s * (t.get("net_pct", 0) / 100)
                curr_w += t["pnl_usd"]
                t["wallet_snapshot"] = curr_w
                comp_trades.append(t)
            recent_trades = comp_trades
        except Exception as e:
            print(f"Main error: {e}")
    print(f"Result length: {len(recent_trades)}")

if __name__ == "__main__":
    test()
