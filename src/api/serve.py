"""
KAT Inference API (V12.0 - Sovereign Authority) ⚓🏛️🧠
======================================================
FastAPI server exposing neural statistics and live expert council.
New in V12.0:
  - /api/health    : Training process liveness check
  - /api/price     : Live BTC price (cached 60s)
  - /api/stats     : Step counter, total_steps, epoch timestamps for real ETR
"""

import os, sys, json, random, asyncio, time, subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

# ── Config Sync ───────────────────────────────────────────────────────────────
SRC_ROOT  = Path(__file__).parent.parent
PROJ_ROOT = SRC_ROOT.parent
LOG_DIR   = PROJ_ROOT / "logs"
sys.path.insert(0, str(SRC_ROOT))

from config.sovereign_config import FEE_RATE, INITIAL_WALLET_USD, POSITION_SIZE_PCT, PROFIT_GOAL_PCT, RISK_MULTIPLIER

# ── Price Cache (avoid hammering exchange every 5s) ───────────────────────────
_price_cache = {"price": 0.0, "ts": 0.0}
_PRICE_TTL   = 60  # seconds

def get_live_btc_price() -> float:
    global _price_cache
    if time.time() - _price_cache["ts"] < _PRICE_TTL and _price_cache["price"] > 0:
        return _price_cache["price"]
    import urllib.request

    # ── Primary: Delta Exchange India ─────────────────────────────────────────
    try:
        url = "https://api.india.delta.exchange/v2/tickers/BTCUSD"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read())
        price = float(data["result"]["close"])
        _price_cache = {"price": price, "ts": time.time(), "source": "delta"}
        return price
    except Exception as e_delta:
        pass  # Fall through to Binance backup

    # ── Fallback: Binance (data source for training — not our exchange) ────────
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        price = float(data["price"])
        _price_cache = {"price": price, "ts": time.time(), "source": "binance_fallback"}
        return price
    except:
        return _price_cache.get("price", 0.0)


# ── Epoch Timing Cache (for real ETR) ─────────────────────────────────────────
_epoch_timestamps: dict = {}  # epoch_num -> unix_timestamp when completed

class SovereignDiscourse:
    def __init__(self):
        self.pool_path = "/var/www/html/ML/kat/src/api/experts_pool.json"
        self.experts = {}
        self.load_experts()

    def load_experts(self):
        try:
            with open(self.pool_path, "r") as f:
                self.experts = json.load(f)
        except:
            # Fallback if file missing
            self.experts = {str(i): {"name": f"Expert #{i}", "role": "General Analyst", "personality": "Neutral"} for i in range(256)}

    def get_experts_by_role(self, roles: list) -> list:
        found = [v for k, v in self.experts.items() if v["role"] in roles]
        return found if found else list(self.experts.values())

    def generate_debate(self, ctx: dict) -> list:
        ist_now  = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%d %b %I:%M %p")
        price    = ctx.get("price",        0.0)
        fee      = ctx.get("fee_rate",  0.0012) * 100
        cert     = ctx.get("certainty",    0.0)
        vloss    = ctx.get("val_loss",     0.0)
        vacc     = ctx.get("val_acc",      0.0) * 100
        epoch    = ctx.get("epoch",          0)
        step     = ctx.get("current_step",   0)
        tstep    = ctx.get("total_steps", 7578)
        wr       = ctx.get("win_rate",     0.0)
        net_pnl  = ctx.get("net_profit",   0.0)
        max_dd   = ctx.get("max_dd",       0.0) * 100
        candles  = ctx.get("candle_count", "400,000")
        
        fee_per_rt = price * fee * 2 / 100

        # Step 1: Identify specialized experts for each part of the debate
        e_market   = random.choice(self.get_experts_by_role(["Macro Sentiment Tracker", "Order Flow Analyst", "Momentum Scalper"]))
        e_econ     = random.choice(self.get_experts_by_role(["Liquidity Provisioner", "Execution Optimization Node", "Slippage Mitigation Lead", "Funding Rate Analyst"]))
        e_neural   = random.choice(self.get_experts_by_role(["Gated Neural Guardian", "Deep Feature Interpreter", "Certainty Threshold Sentry"]))
        e_history  = random.choice(self.get_experts_by_role(["High-Frequency Striker", "Volatility Arbitrageur", "Mean Reversion Specialist"]))
        e_risk     = random.choice(self.get_experts_by_role(["Risk Management Auditor", "Delta-Neutral Hedger", "Gamma-Exposure Monitor"]))
        e_verdict  = random.choice(list(self.experts.values())) # The Lead for this round

        conversation = []

        # 1. Market Analysis
        if cert > 100:
            text = f"Looks like we're all seeing the same setup. Certainty is at {cert:.1f}. I'm proposing we take a position at ${price:,.1f}."
        elif cert > 70:
            text = f"I'm seeing a potential setup near ${price:,.1f}, but the group isn't fully convinced yet. Score is only {cert:.1f}."
        else:
            text = f"The market is too choppy right now around ${price:,.1f}. We're way too divided to take a trade (score: {cert:.1f})."
        conversation.append({"speaker": f"{e_market['name']} [{e_market['role']}]", "text": text})

        # 2. Economic Rebuttal
        if fee_per_rt > 0:
            text = f"I hear you, {e_market['name']}, but as a {e_econ['role']}, I have to flag the fees. We need at least a ${fee_per_rt:.1f} move just to break even."
        else:
            text = f"Since we're simulating, fees are secondary, but {e_market['name']} is right about the liquidity at ${price:,.1f}."
        conversation.append({"speaker": f"{e_econ['name']} [{e_econ['role']}]", "text": text})

        # 3. Neural Check
        if vacc >= 53:
            text = f"Our accuracy is holding strong at {vacc:.2f}%. We have the statistical edge to beat those fees. My gated logic confirms {e_market['name']}'s read."
        elif vacc > 0:
            text = f"Wait, as a {e_neural['role']}, I'm seeing accuracy at only {vacc:.2f}%. We shouldn't force trades until we cross 53%."
        else:
            text = f"The neural weights for Epoch {epoch} aren't ready. We're flying blind without accuracy data."
        conversation.append({"speaker": f"{e_neural['name']} [{e_neural['role']}]", "text": text})

        # 4. Training Update
        if vloss > 0:
            text = f"Agreed. Validation loss is still {vloss:.4f}. The neural weights haven't stabilized enough to trust this setup yet."
        elif step > 0:
            text = f"We are mid-epoch {epoch}, actively crunching step {step:,}/{tstep:,}. Let the model learn before we commit."
        else:
            text = f"Just booting up. Give the gates a minute to process the {candles} historical candles."
        conversation.append({"speaker": f"{e_neural['name']} [{e_neural['role']}]", "text": text})

        # 5. Track Record
        if wr > 0:
            if wr >= 50:
                text = f"Our win rate is {wr:.1f}% and we're up {'+' if net_pnl>0 else ''}{net_pnl:.2f} USD. The strategy works, {e_econ['name']}."
            else:
                text = f"Have you seen the P&L? We're at {'+' if net_pnl>0 else ''}{net_pnl:.2f} USD. Win rate is only {wr:.1f}%. I say we skip."
        else:
            text = "No paper trade history yet. We need a track record before we trust this signal."
        conversation.append({"speaker": f"{e_history['name']} [{e_history['role']}]", "text": text})

        # 6. Risk Veto
        if max_dd > 10:
            text = f"Risk management here. Drawdown is {max_dd:.2f}%. I am vetoing {e_market['name']}'s proposal to protect the remaining balance."
        elif max_dd > 0:
            text = f"Risk checks out. Drawdown is controlled at {max_dd:.2f}%. We have the breathing room to take this setup."
        else:
            text = f"Risk desk is green. No active drawdown, so you're cleared to trade if you're sure, {e_market['name']}."
        conversation.append({"speaker": f"{e_risk['name']} [{e_risk['role']}]", "text": text})

        # 7. Final Verdict
        if cert > 100 and vacc >= 53 and max_dd < 10:
            text = "Alright, the numbers look solid. The council is aligned. Authorizing the trade."
        else:
            text = f"Too many red flags. Verdict: HOLD. Let Epoch {epoch} conclude."
        conversation.append({"speaker": f"{e_verdict['name']} [{e_verdict['role']}]", "text": text})

        debate = []
        for i, msg in enumerate(conversation):
            import hashlib
            # Seed the ID with step and index so it's stable within the same training block
            m_id = int(hashlib.md5(f"{step}-{i}-{msg['text'][:20]}".encode()).hexdigest(), 16) % 10000000
            debate.append({
                "speaker": msg["speaker"],
                "avatar": "🤖",
                "text": msg["text"],
                "time": ist_now,
                "id": m_id
            })

        return debate



discourse_engine = SovereignDiscourse()

# ─────────────────────────────────────────────────────────────────────────────
# LOG PARSER — extracts epochs + live step info
# ─────────────────────────────────────────────────────────────────────────────

def parse_training_log():
    import re
    log_path = LOG_DIR / "iron_oracle_v11.log"
    epochs       = []
    candle_count = "400,000"
    current_step = 0
    total_steps  = 7578   # default
    current_epoch_num = 0

    if not log_path.exists():
        return epochs, candle_count, current_step, total_steps, current_epoch_num

    try:
        with open(log_path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")

        # Candle count
        c_match = re.search(r"Loading ([\d,]+) candles", raw)
        if c_match:
            candle_count = c_match.group(1)

        # Total steps per epoch
        s_match = re.search(r"Steps/epoch:\s*(\d+)", raw)
        if s_match:
            total_steps = int(s_match.group(1))

        # Completed epochs from dashboard table lines
        table_pat = re.compile(r"\d{2}:\d{2}:\d{2}\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")
        seen = {}
        for m in table_pat.finditer(raw):
            ep_int = int(m.group(1))
            if ep_int > 0:
                seen[ep_int] = {"epoch": ep_int, "val_acc": float(m.group(2)),
                                "certainty": float(m.group(3)), "val_loss": 0.0,
                                "completed_at": 0}

        # val_loss from Keras output
        keras_pat = re.compile(r"Epoch (\d+)/300.*?val_loss:\s*([\d.]+)", re.DOTALL)
        for m in keras_pat.finditer(raw):
            ep_int = int(m.group(1))
            if ep_int in seen:
                seen[ep_int]["val_loss"] = float(m.group(2))

        # Epoch completion timestamps (for real ETR)
        time_pat = re.compile(r"(\d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|")
        for m in time_pat.finditer(raw):
            ep_int = int(m.group(2))
            if ep_int in seen:
                try:
                    t = datetime.now().replace(
                        hour=int(m.group(1)[:2]),
                        minute=int(m.group(1)[3:5]),
                        second=int(m.group(1)[6:8]),
                        microsecond=0
                    )
                    seen[ep_int]["completed_at"] = t.timestamp()
                    _epoch_timestamps[ep_int] = t.timestamp()
                except: pass

        epochs = [seen[ep] for ep in sorted(seen)]

        # Current live step from last Keras progress line
        # Handle ANSI color codes that might be surrounding the numbers
        step_pat = re.compile(r"(\d+)/(\d+).*?[━=\-]{2,}")
        all_steps = step_pat.findall(raw)
        if all_steps:
            last = all_steps[-1]
            current_step  = int(last[0])
            total_steps   = int(last[1])

        # Current epoch (last "Epoch X/300" seen)
        ep_header = re.compile(r"Epoch (\d+)/300")
        ep_headers = ep_header.findall(raw)
        if ep_headers:
            current_epoch_num = int(ep_headers[-1])

    except:
        pass

    return epochs, candle_count, current_step, total_steps, current_epoch_num

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="KAT Sovereign Console V12.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    dash_path = Path(__file__).parent / "dashboard.html"
    if dash_path.exists():
        return dash_path.read_text()
    return "<h1>Dashboard Missing</h1>"

# ── NEW: Health endpoint ──────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Check if training process is alive."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py"],
            capture_output=True, text=True
        )
        training_alive = bool(result.stdout.strip())
        server_alive   = True
        return {
            "training": training_alive,
            "server": server_alive,
            "pid": result.stdout.strip() if training_alive else None,
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {"training": False, "server": True, "pid": None}

# ── NEW: Live price endpoint ──────────────────────────────────────────────────
@app.get("/api/price")
async def live_price():
    """Return live BTC price — Delta Exchange India primary, Binance fallback."""
    price = get_live_btc_price()
    return {
        "price":     price,
        "symbol":    "BTCUSD",
        "source":    _price_cache.get("source", "unknown"),
        "cached_at": datetime.fromtimestamp(_price_cache["ts"]).isoformat() if _price_cache["ts"] > 0 else None
    }

# ── Reset wallet ──────────────────────────────────────────────────────────────
@app.post("/api/reset_wallet")
def reset_wallet():
    try:
        reset_file = LOG_DIR / "sim_reset.txt"
        with open(reset_file, "w") as f:
            f.write(datetime.now().isoformat())
        return JSONResponse({"status": "success", "message": "Wallet recapitalized."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ── Main stats endpoint ───────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    epochs, candle_count, current_step, total_steps, current_epoch_num = parse_training_log()
    latest = epochs[-1] if epochs else {}

    # Average epoch duration for real ETR
    avg_epoch_secs = None
    if len(_epoch_timestamps) >= 2:
        sorted_eps = sorted(_epoch_timestamps.items())
        diffs = [sorted_eps[i+1][1] - sorted_eps[i][1] for i in range(len(sorted_eps)-1)]
        avg_epoch_secs = float(np.mean(diffs)) if diffs else None

    # 1. ROI & Expert Heatmap
    expert_heatmap = [random.random() * 0.2 for _ in range(256)]
    roi_path = LOG_DIR / "latest_roi.json"
    roi_block = {"net": None, "trades": None, "status": "measure",
                 "note": f"FEE GATE: {FEE_RATE*100:.2f}%", "wallet": INITIAL_WALLET_USD}

    if roi_path.exists():
        try:
            with open(roi_path, "r") as f:
                rd = json.load(f)
            t80 = rd.get("tiers", {}).get("80")
            if t80:
                roi_block.update({"net": t80["net"], "trades": t80["trades"], "status": "live"})
            if "expert_heatmap" in rd:
                expert_heatmap = rd["expert_heatmap"]
            # Build expert_fight for certainty chart from ROI tiers
            if "tiers" in rd:
                roi_block["expert_fight"] = [
                    rd["tiers"].get(str(t), {}).get("win_rate", 0)
                    for t in [60, 65, 70, 75, 80, 85, 90, 95]
                ]
        except: pass

    # 2. Sim Trades with compounding wallet
    recent_trades  = []
    latest_wallet  = INITIAL_WALLET_USD
    trades_path    = LOG_DIR / "recent_sim_trades.json"
    if trades_path.exists():
        try:
            with open(trades_path, "r") as f:
                raw_trades = json.load(f)
            curr_w   = INITIAL_WALLET_USD
            comp_trades = []
            for t in reversed(raw_trades):
                try:
                    utc_t = datetime.fromisoformat(t["timestamp"])
                    ist_t = utc_t + timedelta(hours=5, minutes=30)
                    t["timestamp"] = ist_t.strftime("%d %b %Y %I:%M %p")
                except: pass

                pos_s          = curr_w * POSITION_SIZE_PCT
                t["value_usd"] = pos_s
                t["pnl_usd"]   = pos_s * (t.get("net_pct", 0) / 100)
                curr_w        += t["pnl_usd"]
                t["wallet_snapshot"] = curr_w

                if "expert" not in t:
                    import hashlib
                    seed = int(hashlib.md5(f"{t.get('timestamp','')}{t.get('entry',0)}".encode()).hexdigest(), 16)
                    exp_id = str(seed % 256)
                    exp_data = discourse_engine.experts.get(exp_id, {"name": f"Expert #{exp_id}", "role": "Analyst"})
                    t["expert"] = f"{exp_data['name']} [{exp_data['role']}]"
                
                t["avatar"] = "🤖"
                comp_trades.append(t)

            recent_trades = comp_trades
            latest_wallet = curr_w
        except Exception as e:
            print(f"DEBUG: Trade loading failed: {e}")
            pass


    # 3. Quant Risk Metrics (Needed for Debate)
    max_dd = sharpe = profit_factor = win_rate = 0.0
    if recent_trades:
        sorted_trades = sorted(recent_trades, key=lambda x: x["timestamp"])
        returns       = [t["net_pct"] / 100 for t in sorted_trades]
        peak = curr_w = INITIAL_WALLET_USD
        drawdowns = []
        for r in returns:
            curr_w *= (1 + r)
            if curr_w > peak: peak = curr_w
            drawdowns.append((peak - curr_w) / peak)
        max_dd = max(drawdowns) if drawdowns else 0.0
        if len(returns) > 1:
            avg_r  = np.mean(returns)
            std_r  = np.std(returns) + 1e-9
            sharpe = (avg_r / std_r) * np.sqrt(35040)
        gains  = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        win_rate      = (len(gains) / len(returns)) * 100
        profit_factor = sum(gains) / (sum(losses) + 1e-9)

    # 4. Dialogue (Real Data Context)
    latest_price   = _price_cache.get("price", 0.0) or (recent_trades[0].get("entry", 0) if recent_trades else 0)
    certainty      = latest.get("certainty", 0.5) if latest else 0.5
    active_indices = sorted(range(len(expert_heatmap)), key=lambda i: expert_heatmap[i], reverse=True)[:10]
    dialogue       = discourse_engine.generate_debate({
        "price":        latest_price,
        "fee_rate":     FEE_RATE,
        "certainty":    certainty,
        "val_loss":     latest.get("val_loss", 0.0) if latest else 0.0,
        "val_acc":      latest.get("val_acc", 0.0) if latest else 0.0,
        "epoch":        current_epoch_num,
        "total_epochs": 300,
        "current_step": current_step,
        "total_steps":  total_steps,
        "win_rate":     win_rate,
        "net_profit":   latest_wallet - INITIAL_WALLET_USD,
        "last_pnl":     recent_trades[0].get("net_pct", 0.0) if recent_trades else 0.0,
        "last_side":    recent_trades[0].get("side", "—") if recent_trades else "—",
        "max_dd":       max_dd,
        "candle_count": candle_count,
        "active_indices": active_indices
    })

    return {
        "status":           "TRAINING" if epochs or current_step > 0 else "IDLE",
        "epochs":           epochs,
        "latest":           latest,
        "total_epochs":     300,
        "current_epoch":    current_epoch_num,
        "current_step":     current_step,
        "total_steps":      total_steps,
        "avg_epoch_secs":   avg_epoch_secs,
        "certainty":        certainty,
        "val_loss":         latest.get("val_loss", 0.0) if latest else 0.0,
        "val_acc":          latest.get("val_acc", 0.0) if latest else 0.0,
        "roi":              roi_block,
        "net_profit":       latest_wallet - INITIAL_WALLET_USD,
        "trades":           recent_trades,
        "dialogue":         dialogue,
        "expert_heatmap":   expert_heatmap,
        "risk": {
            "max_dd":        max_dd,
            "lev_health":    max(0, 100 - (max_dd * RISK_MULTIPLIER)),
            "sharpe":        sharpe,
            "profit_factor": profit_factor,
            "win_rate":      win_rate
        },

        "config": {
            "fee_rate":      FEE_RATE,
            "initial_wallet": INITIAL_WALLET_USD,
            "wallet":        latest_wallet,
            "pos_pct":       POSITION_SIZE_PCT,
            "profit_goal":   PROFIT_GOAL_PCT,
            "risk_mult":     RISK_MULTIPLIER
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)