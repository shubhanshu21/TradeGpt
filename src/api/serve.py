"""
KAT Inference API (V11.2 - Sovereign Authority) ⚓🏛️🧠
======================================================
FastAPI server exposing neural statistics and live expert council.
"""

import os, sys, json, random, asyncio, time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# ── Config Sync ──────────────────────────────────────────────────────────────
SRC_ROOT  = Path(__file__).parent.parent
PROJ_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from config.sovereign_config import FEE_RATE, INITIAL_WALLET_USD, POSITION_SIZE_PCT

# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def parse_training_log():
    import re
    log_path = PROJ_ROOT / "logs" / "iron_oracle_v11.log"
    epochs = []
    if not log_path.exists(): return epochs
    try:
        with open(log_path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        table_pat = re.compile(r"\d{2}:\d{2}:\d{2}\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")
        seen = {}
        for m in table_pat.finditer(raw):
            ep_int = int(m.group(1))
            if ep_int > 0:
                seen[ep_int] = {"epoch": ep_int, "val_acc": float(m.group(2)), "certainty": float(m.group(3)), "val_loss": 0.0}
        keras_pat = re.compile(r"Epoch (\d+)/300.*?val_loss:\s*([\d.]+)", re.DOTALL)
        for m in keras_pat.finditer(raw):
            ep_int = int(m.group(1))
            if ep_int in seen: seen[ep_int]["val_loss"] = float(m.group(2))
        epochs = [seen[ep] for ep in sorted(seen)]
    except: pass
    return epochs

# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="KAT Sovereign Console")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    dash_path = Path(__file__).parent / "dashboard.html"
    if dash_path.exists(): return dash_path.read_text()
    return "<h1>KAT Dashboard Missing</h1>"

@app.get("/api/stats")
async def get_stats():
    epochs = parse_training_log()
    latest = epochs[-1] if epochs else {}
    
    # 1. Load ROI & Expert Heatmap
    roi_net, roi_trades, roi_status = None, None, "measure"
    expert_heatmap = [random.random() * 0.2 for _ in range(256)]
    
    roi_path = PROJ_ROOT / "logs" / "latest_roi.json"
    if roi_path.exists():
        try:
            with open(roi_path, "r") as f:
                rd = json.load(f)
            t80 = rd["tiers"].get("80")
            if t80:
                roi_net = t80["net"]
                roi_trades = t80["trades"]
                roi_status = "live"
            if "expert_heatmap" in rd:
                expert_heatmap = rd["expert_heatmap"]
        except: pass

    # 2. Compounding Wallet Logic (Neural Truth V11.2)
    recent_trades = []
    latest_wallet = INITIAL_WALLET_USD
    trades_path = PROJ_ROOT / "logs" / "recent_sim_trades.json"
    if trades_path.exists():
        try:
            with open(trades_path, "r") as f:
                raw_trades = json.load(f)
            prefixes = ["Neon", "Volt", "Cipher", "Ghost", "Logic", "Vector", "Pulse", "Neural", "Cyber", "Quant", "Delta", "Gamma", "Alpha", "Zenith", "Apex", "Flow"]
            suffixes = ["Hunter", "Scout", "Tracker", "Oracle", "Sentinel", "Core", "Node", "Gate", "Shell", "Link", "Edge", "Vortex", "Matrix", "System", "Prime", "Zero"]
            
            curr_w = INITIAL_WALLET_USD
            comp_trades = []
            for t in reversed(raw_trades):
                try:
                    # Parse ISO format from train.py (V11.2 Precision)
                    utc_t = datetime.fromisoformat(t["timestamp"])
                    ist_t = utc_t + timedelta(hours=5, minutes=30)
                    t["timestamp"] = ist_t.strftime("%d %b %Y %I:%M %p")
                except: pass
                
                # Dynamic Position Sizing
                pos_s = curr_w * POSITION_SIZE_PCT
                t["value_usd"] = pos_s
                t["pnl_usd"]   = pos_s * (t.get("net_pct", 0) / 100)
                
                # Compound
                curr_w += t["pnl_usd"]
                t["wallet_snapshot"] = curr_w
                
                t["expert"] = f"{random.choice(prefixes)}-{random.choice(suffixes)} #{random.randint(0, 255):03d}"
                t["avatar"] = "🤖"
                comp_trades.append(t)
            
            recent_trades = comp_trades[::-1]
            latest_wallet = curr_w
        except: pass

    roi_block = {
        "net": roi_net, 
        "trades": roi_trades, 
        "status": roi_status, 
        "note": f"FEE GATE: {FEE_RATE*100:.2f}%",
        "wallet": latest_wallet
    }

    # 3. Neural Dialogue Engine
    dialogue = []
    try:
        active_indices = sorted(range(len(expert_heatmap)), key=lambda i: expert_heatmap[i], reverse=True)[:10]
        latest_price = latest.get("price", 0) if latest else 0
        trend_label = "bullish" if (recent_trades and recent_trades[0].get("net_pct", 0) > 0) else "choppy"
        ist_now = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%I:%M %p")
        
        leads = {"Sentinel": "🛡️", "Hunter": "🦁", "Oracle": "🦉", "Tracker": "🐋", "Scout": "🐙"}
        def get_expert():
            idx = random.choice(active_indices)
            return f"{random.choice(prefixes)}-{random.choice(suffixes)} #{idx:03d}"
        def inject_mentions(text, current_speaker):
            target = random.choice(list(leads.keys()) + [get_expert()])
            return text.replace("{expert}", target)

        if trend_label == "bullish":
            pools = [
                [
                    {"speaker": "Hunter", "avatar": "🦁", "text": "BLOOD IN THE WATER! [EUPHORIC] The bulls are breaking the cage. Buy the spike!"},
                    {"speaker": get_expert(), "avatar": "🤖", "text": "Neural link stable. [CONFIDENT] Confirming Hunter. Market sentiment is reaching fever pitch."},
                    {"speaker": "Oracle", "avatar": "🦉", "text": f"Patience, Hunter. [CALM] You're smelling greed, not trend. Tape is thin above ${latest_price:,.0f}."},
                    {"speaker": get_expert(), "avatar": "🤖", "text": "Scaning liquidity. [NERVOUS] Oracle is right. Massive sell-wall hiding at the gate."},
                    {"speaker": "Sentinel", "avatar": "🛡️", "text": "IT'S A TRAP! [AGITATED] {expert}, if we enter here, we're the liquidity. Abort!"},
                    {"speaker": "Oracle", "avatar": "🦉", "text": "Decision reached. [RESOLUTE] {expert} has the data. We wait for the shakeout."}
                ]
            ]
        else:
            pools = [
                [
                    {"speaker": get_expert(), "avatar": "🤖", "text": f"Negative, Hunter. [STERN] The fee gate is {FEE_RATE*100:.2f}%. Strategy: HOLD."},
                    {"speaker": "Oracle", "avatar": "🦉", "text": "Deadlock is its own lesson, {expert}. [TEACHING] Only fools trade in a cemetery."},
                    {"speaker": "Sentinel", "avatar": "🛡️", "text": "Listen to the bot, Hunter. [ANNOYED] {expert}, sit on your hands."},
                    {"speaker": "Oracle", "avatar": "🦉", "text": "Deadlock confirmed. [FINAL] {expert}, stay liquid. War starts tomorrow."}
                ]
            ]
        
        seed_idx = int(time.time() / 60) % len(pools)
        raw_dialogue = pools[seed_idx]
        for m in raw_dialogue:
            m["text"] = inject_mentions(m["text"], m["speaker"])
            m["time"] = ist_now
            dialogue.append(m)
    except: pass

    # 4. Quant Metrics Engine (Institutional Grade V11.2)
    max_dd = 0.0
    sharpe = 0.0
    profit_factor = 0.0
    win_rate = 0.0
    
    if recent_trades:
        # Sort chronologically for MDD/Sharpe
        sorted_trades = sorted(recent_trades, key=lambda x: x["timestamp"])
        returns = [t["net_pct"] / 100 for t in sorted_trades]
        
        # Max Drawdown
        peak = INITIAL_WALLET_USD
        curr_w = INITIAL_WALLET_USD
        drawdowns = []
        for t in returns:
            curr_w *= (1 + t)
            if curr_w > peak: peak = curr_w
            dd = (peak - curr_w) / peak
            drawdowns.append(dd)
        max_dd = max(drawdowns) if drawdowns else 0.0
        
        # Sharpe Ratio (Risk-Adjusted)
        if len(returns) > 1:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns) + 1e-9
            # Annualized (Assuming 15m timeframe -> ~35,000 candles/year)
            sharpe = (avg_ret / std_ret) * np.sqrt(35040)
            
        # Profit Factor & Win Rate
        gains = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        win_rate = (len(gains) / len(returns)) * 100
        profit_factor = sum(gains) / (sum(losses) + 1e-9)

    return {
        "status": "TRAINING" if epochs else "IDLE",
        "latest": latest,
        "roi": roi_block,
        "net_profit": latest_wallet - INITIAL_WALLET_USD,
        "trades": recent_trades,
        "dialogue": dialogue,
        "expert_heatmap": expert_heatmap,
        "risk": {
            "max_dd": max_dd,
            "lev_health": 100 - (max_dd * RISK_MULTIPLIER),
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "win_rate": win_rate
        },
        "config": {
            "fee_rate": FEE_RATE, 
            "initial_wallet": INITIAL_WALLET_USD, 
            "wallet": latest_wallet, 
            "pos_pct": POSITION_SIZE_PCT,
            "profit_goal": PROFIT_GOAL_PCT,
            "risk_mult": RISK_MULTIPLIER
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)