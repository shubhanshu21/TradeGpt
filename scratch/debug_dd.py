import json
import numpy as np
from pathlib import Path

LOG_DIR = Path("/var/www/html/ML/kat/logs")
trades_path = LOG_DIR / "recent_sim_trades.json"

if trades_path.exists():
    with open(trades_path, "r") as f:
        all_trades = json.load(f)
    
    if all_trades:
        pnls = [t["net_pct"] / 100.0 for t in all_trades]
        
        # Chronological order
        chron_pnls = list(reversed(pnls))
        
        equity = [1.0]
        for p in chron_pnls:
            equity.append(equity[-1] * (1.0 + p))
            
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)
        
        print(f"Total Trades: {len(all_trades)}")
        print(f"Pnls (first 5): {chron_pnls[:5]}")
        print(f"Equity (first 5): {equity[:5].tolist()}")
        print(f"Peaks (first 5): {peak[:5].tolist()}")
        print(f"Drawdowns (first 5): {drawdown[:5].tolist()}")
        print(f"MAX DD: {max_dd * 100:.4f}%")
    else:
        print("No trades found.")
else:
    print("Trades file not found.")
