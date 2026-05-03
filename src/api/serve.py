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

from config.sovereign_config import FEE_RATE, INITIAL_WALLET_USD, POSITION_SIZE_PCT, PROFIT_GOAL_PCT, RISK_MULTIPLIER

class SovereignDiscourse:
    """Institutional Neural Debate Engine (V12.0) — Context-Aware Discourse Generation."""
    def __init__(self):
        self.leads = {"Oracle": "🦉", "Hunter": "🦁", "Sentinel": "🛡️", "Tracker": "🐋", "Scout": "🐙"}
        self.prefixes = ["Neon", "Volt", "Cipher", "Ghost", "Logic", "Vector", "Pulse", "Neural", "Cyber", "Quant", "Delta", "Gamma", "Alpha", "Zenith", "Apex", "Flow"]
        self.suffixes = ["Hunter", "Scout", "Tracker", "Oracle", "Sentinel", "Core", "Node", "Gate", "Shell", "Link", "Edge", "Vortex", "Matrix", "System", "Prime", "Zero"]
        self.moods = ["[CALM]", "[AGGRESSIVE]", "[STERN]", "[EUPHORIC]", "[NERVOUS]", "[RESOLUTE]", "[TEACHING]", "[ANALYTICAL]"]

    def get_expert(self, active_indices):
        idx = random.choice(active_indices) if active_indices else random.randint(0, 255)
        return f"{random.choice(self.prefixes)}-{random.choice(self.suffixes)} #{idx:03d}", "🤖"

    def generate_debate(self, latest_price, fee_rate, certainty, active_indices, candle_count="400,000"):
        ist_now = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%I:%M %p")
        
        # Simple Indian English Personality Templates
        idle_templates = [
            "Network is connecting. {mood} Please wait for the mission to start.",
            "Checking all systems. {mood} 256 Experts are ready on standby.",
            "Everything is green. {mood} We are waiting for the market signal.",
            "Model is cooling down. {mood} Ready to check {candles} data points."
        ]
        oracle_templates = [
            "Price is holding steady at {price}. {mood} Keep watching carefully.",
            "Market is a bit confused right now. {mood} Strategy: Be patient and disciplined.",
            "I feel people are getting greedy. {mood} Selling can happen above {price}.",
            "History says it might pull back. {mood} Wait for a better entry point."
        ]
        hunter_templates = [
            "CHANCE IS HERE! {mood} Buy the breakout at {price} now!",
            "Bulls are very strong today! {mood} Let's push the neural gates!",
            "Momentum is looking very good! {mood} Don't miss this move!",
            "I see a big opportunity. {mood} This is the time to be a hunter."
        ]
        sentinel_templates = [
            "BE CAREFUL! {mood} {expert}, the fee is {fee}% — don't lose money.",
            "Saving capital is most important. {mood} Don't trade in this sideways market.",
            "Risk team says stop here. {mood} Cancel the entry for now!",
            "Safety protocols are on. {mood} We will wait until the risk is less."
        ]
        tracker_templates = [
            "Checking who is buying... {mood} Big orders detected at {price}.",
            "Big players are moving money. {mood} They are trying to trap small traders.",
            "Buying pressure is high! {mood} Market depth is changing fast.",
            "Following the big money flow. {mood} Best entry is near {price}."
        ]

        debate = []
        # Construct an 8-message thread for high density
        participants = ["Oracle", "Hunter", "Sentinel", "Tracker"] * 2
        random.shuffle(participants)
        
        for p in participants[:8]:
            role = p
            avatar = self.leads[p]
            mood = random.choice(self.moods)
            expert, _ = self.get_expert(active_indices)
            
            # Switch to IDLE templates if no price data or very low price
            if float(latest_price) < 100:
                tpl = random.choice(idle_templates)
            else:
                if role == "Oracle":   tpl = random.choice(oracle_templates)
                elif role == "Hunter": tpl = random.choice(hunter_templates)
                elif role == "Sentinel": tpl = random.choice(sentinel_templates)
                else:                  tpl = random.choice(tracker_templates)
            
            text = tpl.format(
                price=f"${latest_price:,.2f}", 
                fee=f"{fee_rate*100:.2f}",
                cert=f"{certainty*100:.1f}",
                mood=mood,
                expert=expert,
                candles=candle_count
            )
            debate.append({"speaker": role, "avatar": avatar, "text": text, "time": ist_now})
            
        return debate

discourse_engine = SovereignDiscourse()

# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def parse_training_log():
    import re
    log_path = PROJ_ROOT / "logs" / "iron_oracle_v11.log"
    epochs = []
    candle_count = "400,000" # Safe default
    if not log_path.exists(): return epochs, candle_count
    try:
        with open(log_path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        
        # Extract Candle Count
        c_match = re.search(r"Loading ([\d,]+) candles", raw)
        if c_match: candle_count = c_match.group(1)
        
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
    return epochs, candle_count

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

from fastapi.responses import JSONResponse
@app.post("/api/reset_wallet")
def reset_wallet():
    try:
        reset_file = LOG_DIR / "sim_reset.txt"
        with open(reset_file, "w") as f:
            f.write(datetime.now().isoformat())
        return JSONResponse({"status": "success", "message": "Wallet recapitalized. Simulation slate wiped."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/stats")
async def get_stats():
    epochs, candle_count = parse_training_log()
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
            
            recent_trades = comp_trades
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
    latest_price = latest.get("price", 0) if latest else 0
    if not latest_price and recent_trades:
        latest_price = recent_trades[0].get("entry", 0)
        
    certainty = latest.get("certainty", 0.5) if latest else 0.5
    active_indices = sorted(range(len(expert_heatmap)), key=lambda i: expert_heatmap[i], reverse=True)[:10]
    dialogue = discourse_engine.generate_debate(latest_price, FEE_RATE, certainty, active_indices, candle_count)


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