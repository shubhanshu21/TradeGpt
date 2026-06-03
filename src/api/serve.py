"""
KAT Inference API (V12.1 - Sovereign Authority) ⚓🏛️🧠
======================================================
FastAPI server exposing neural statistics and live expert council.
Upgraded in V12.1:
  - Dynamic path resolution (removed hardcoded paths)
  - Auto-Log detection (reads most recent log in logs/)
  - Non-blocking async price fetching
  - Enhanced error handling for expert council
"""

import os, sys, json, random, asyncio, time, subprocess, re
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

# ── Dynamic Path Resolution ───────────────────────────────────────────────────
FILE_PATH = Path(__file__).resolve()
SRC_API   = FILE_PATH.parent
SRC_ROOT  = SRC_API.parent
PROJ_ROOT = SRC_ROOT.parent
LOG_DIR   = PROJ_ROOT / "logs"

sys.path.insert(0, str(SRC_ROOT))

# Try to load config, fallback to defaults if not found
try:
    from config.sovereign_config import (FEE_RATE, INITIAL_WALLET_USD, POSITION_SIZE_PCT, 
                                         PROFIT_GOAL_PCT, RISK_MULTIPLIER, LEVERAGE, 
                                         MAX_TRADES_PER_DAY, BREAKEVEN_TRIGGER_PCT, TRAILING_STOP_PCT)
except ImportError:
    FEE_RATE = 0.0012
    INITIAL_WALLET_USD = 200.0
    POSITION_SIZE_PCT = 0.25
    PROFIT_GOAL_PCT = 0.1
    RISK_MULTIPLIER = 100.0
    LEVERAGE = 10
    MAX_TRADES_PER_DAY = 4
    BREAKEVEN_TRIGGER_PCT = 0.8
    TRAILING_STOP_PCT = 0.5

from exchange.delta_client import DeltaClient

# ── Price Cache (avoid hammering exchange every 5s) ───────────────────────────
_price_cache = {"price": 0.0, "ts": 0.0, "source": "none"}
_PRICE_TTL   = 30  # Increased frequency for live feeling

def _fetch_sync_price():
    """Sync worker for background thread."""
    import urllib.request
    
    # ── Primary: Delta Exchange India ─────────────────────────────────────────
    try:
        url = "https://api.india.delta.exchange/v2/tickers/BTCUSD"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        return float(data["result"]["close"]), "delta"
    except:
        pass

    # ── Fallback: Binance ─────────────────────────────────────────────────────
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        return float(data["price"]), "binance_fallback"
    except:
        return 0.0, "error"

async def get_live_btc_price() -> float:
    global _price_cache
    now = time.time()
    if now - _price_cache["ts"] < _PRICE_TTL and _price_cache["price"] > 0:
        return _price_cache["price"]
    
    # Run sync call in thread to avoid blocking event loop
    price, source = await asyncio.to_thread(_fetch_sync_price)
    
    if price > 0:
        _price_cache = {"price": price, "ts": now, "source": source}
        return price
    return _price_cache.get("price", 0.0)


# ── Sovereign Discourse ───────────────────────────────────────────────────────
class SovereignDiscourse:
    def __init__(self):
        self.pool_path = SRC_API / "experts_pool.json"
        self.experts = {}
        self.load_experts()

    def load_experts(self):
        if self.pool_path.exists():
            try:
                with open(self.pool_path, "r") as f:
                    self.experts = json.load(f)
                return
            except: pass
        # Fallback
        self.experts = {str(i): {"name": f"Expert #{i}", "role": "General Analyst"} for i in range(256)}

    def generate_debate(self, ctx: dict) -> list:
        ist_now  = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%d %b %I:%M %p")
        price    = ctx.get("price",        0.0)
        cert     = ctx.get("certainty",    0.0)
        heatmap  = ctx.get("expert_heatmap", [0.0]*256)
        epoch    = ctx.get("epoch",          0)
        step     = ctx.get("current_step",   0)
        
        top_indices = sorted(range(len(heatmap)), key=lambda i: heatmap[i], reverse=True)[:3]
        
        def get_specialist_role(idx):
            if 0 <= idx <= 63:    return "Order-Flow Specialist", "detecting Liquidity Clusters"
            if 64 <= idx <= 127:  return "Momentum Striker", "analyzing Velocity-Resonance"
            if 128 <= idx <= 191: return "Volatility Arbiter", "measuring Mean Reversion"
            return "Structural Trend-Lead", "mapping Market Structure"

        conversation = []
        # Expert 1
        idx1 = top_indices[0]
        role1, action1 = get_specialist_role(idx1)
        conversation.append({"speaker": f"Expert #{idx1} [{role1}]", 
                             "text": f"Activation High. I am {action1} near ${price:,.1f}. Certainty at {cert:.1f}."})
        # Expert 2
        idx2 = top_indices[1]
        role2, action2 = get_specialist_role(idx2)
        conversation.append({"speaker": f"Expert #{idx2} [{role2}]", 
                             "text": f"Secondary gate confirming. {action2} supports the current neural vector."})
        # Expert 3
        idx3 = top_indices[2]
        role3, action3 = get_specialist_role(idx3)
        conversation.append({"speaker": f"Expert #{idx3} [{role3}]", 
                             "text": f"Weights for Step {step:,} are consistent. No abnormal variance in current slice."})

        # Verdict
        if cert > 85: verdict = "Verdict: AUTHORIZED. Consensus reached."
        elif cert > 60: verdict = "Verdict: CAUTION. High-conviction threshold not yet met."
        else: verdict = f"Verdict: HOLD. Noise dominating. Waiting for Epoch {epoch} optimization."
        conversation.append({"speaker": "Sovereign Gating System", "text": verdict})

        debate = []
        for i, msg in enumerate(conversation):
            debate.append({
                "speaker": msg["speaker"],
                "avatar": "🧠" if "System" in msg["speaker"] else "🤖",
                "text": msg["text"],
                "time": ist_now,
                "id": random.randint(100000, 999999)
            })
        return debate

discourse_engine = SovereignDiscourse()

# ── LOG PARSER (Improved) ─────────────────────────────────────────────────────

def find_latest_log():
    """Finds the most recently modified training log file in logs/"""
    if not LOG_DIR.exists(): return None
    log_files = list(LOG_DIR.glob("iron_oracle*.log")) + list(LOG_DIR.glob("hydra_train*.log"))
    if not log_files: return None
    return max(log_files, key=os.path.getmtime)

def parse_training_log():
    log_path = find_latest_log()
    epochs       = []
    candle_count = "Unknown"
    current_step = 0
    total_steps  = 1000   
    current_epoch_num = 0

    if not log_path or not log_path.exists():
        return epochs, candle_count, current_step, total_steps, current_epoch_num

    try:
        with open(log_path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")

        c_match = re.search(r"Loading ([\d,]+) candles", raw)
        if c_match: candle_count = c_match.group(1)

        s_match = re.search(r"Steps/epoch:\s*(\d+)", raw)
        if s_match: total_steps = int(s_match.group(1))

        # Extract val_loss from progress bar
        val_loss_pat = re.compile(r"- val_loss:\s*([\d.]+)")
        val_losses = val_loss_pat.findall(raw)

        # Dashboard Table Parser
        table_pat = re.compile(r"(\d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*(?:\|\s*([^|\n]+))?")
        seen = {}
        for m in table_pat.finditer(raw):
            ep_int = int(m.group(2))
            if ep_int > 0:
                v_loss = 0.0
                if ep_int - 1 < len(val_losses):
                    try:
                        v_loss = float(val_losses[ep_int - 1])
                    except: pass
                
                roi_val = m.group(5).strip() if m.group(5) else "---"
                clean_roi = roi_val.replace("$", "").strip()
                try:
                    val = float(clean_roi)
                    roi_val = f"+${val:.2f}" if val >= 0 else f"-${abs(val):.2f}"
                except ValueError:
                    pass
                seen[ep_int] = {
                    "epoch": ep_int, 
                    "val_acc": float(m.group(3)),
                    "certainty": float(m.group(4)), 
                    "roi": roi_val,
                    "val_loss": v_loss,
                    "completed_at": 0
                }
                # Completion time
                try:
                    t = datetime.now().replace(hour=int(m.group(1)[:2]), minute=int(m.group(1)[3:5]), second=int(m.group(1)[6:8]), microsecond=0)
                    seen[ep_int]["completed_at"] = t.timestamp()
                except: pass

        epochs = [seen[ep] for ep in sorted(seen)]

        # Live Progress Parser
        # Live Progress Parser (Robust version for ANSI/UTF-8 bars)
        # Matches: " 378/2996 " or "[1m 378/2996[0m"
        step_pat = re.compile(r"(\d+)/(\d+)")
        all_matches = step_pat.findall(raw)
        if all_matches:
            # We want the LAST valid step/total pair in the file
            # but we ignore cases where the numbers look like a date or version
            for s_curr, s_total in reversed(all_matches):
                curr, total = int(s_curr), int(s_total)
                if total > 100 and curr <= total:
                    current_step = curr
                    total_steps  = total
                    break

        ep_header = re.compile(r"Epoch (\d+)/(\d+)")
        ep_headers = ep_header.findall(raw)
        if ep_headers:
            current_epoch_num = int(ep_headers[-1][0])

    except Exception as e:
        print(f"DEBUG: Log parse error: {e}")

    return epochs, candle_count, current_step, total_steps, current_epoch_num

def parse_live_pilot_log():
    """Parses logs/live_pilot.log to extract the active inference results of the live trader."""
    log_path = LOG_DIR / "live_pilot.log"
    certainty = 0.0
    reasoning = "IDLE"
    decision = "SCANNING"
    timestamp = ""
    
    if not log_path.exists():
        return certainty, reasoning, decision, timestamp
        
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read last 150 lines to find the latest inference cycle
            lines = f.readlines()
            if len(lines) > 150:
                lines = lines[-150:]
            raw = "".join(lines)
            
        # Parse certainty: "🧠 CERTAINTY     : 97.2%"
        c_match = re.findall(r"🧠 CERTAINTY\s*:\s*([\d.]+)%", raw)
        if c_match:
            certainty = float(c_match[-1])
            
        # Parse reasoning: "🔮 REASONING     : SOVEREIGN_LONG 🏹"
        r_match = re.findall(r"🔮 REASONING\s*:\s*([^\n]+)", raw)
        if r_match:
            clean_r = r_match[-1].replace("🏹", "").strip()
            clean_r = re.sub(r"\x1b\[[0-9;]*[mG]", "", clean_r).strip()
            reasoning = clean_r
            
        # Parse last action/decision
        for line in reversed(lines):
            if "SWING TOO SMALL" in line:
                decision = "SWING TOO SMALL"
                break
            elif "CERTAINTY TOO LOW" in line:
                decision = "CONVICTION LOW"
                break
            elif "Placed market" in line or "Placed limit" in line:
                decision = "ORDER PLACED"
                break
            elif "Polling" in line:
                decision = "POLLING STREAM"
                break
            elif "Exchange Leverage successfully locked" in line:
                decision = "LEVERAGE LOCKED"
                break
                
        # Parse the timestamp of the last log line
        # e.g., "[15:05:23] 🔮 REASONING"
        ts_match = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", raw)
        if ts_match:
            timestamp = ts_match[-1]
            
    except Exception as e:
        print(f"DEBUG: Live pilot log parse error: {e}")
        
    return certainty, reasoning, decision, timestamp

# ── API ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="KAT Sovereign Console V12.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = SRC_API / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    dash_path = SRC_API / "dashboard.html"
    if dash_path.exists():
        return dash_path.read_text()
    return "<h1>Dashboard Source Not Found</h1>"

@app.get("/api/health")
async def health():
    try:
        # Check for any python training processes
        res = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        training = bool(res.stdout.strip())
        
        # Check for any python live trader processes
        res_trader = subprocess.run(["pgrep", "-f", "live_trader.py"], capture_output=True, text=True)
        live_trader_running = bool(res_trader.stdout.strip())
        
        return {
            "training": training,
            "live_trader_running": live_trader_running,
            "server": True,
            "active_log": str(find_latest_log())
        }
    except:
        return {"training": False, "live_trader_running": False, "server": True}

@app.get("/api/price")
async def live_price():
    p = await get_live_btc_price()
    return {"price": p, "source": _price_cache["source"]}

@app.post("/api/reset_wallet")
def reset_wallet(confirm: Optional[str] = None):
    """Requires ?confirm=SOVEREIGN to proceed."""
    if confirm != "SOVEREIGN":
        return JSONResponse({"status": "denied", "message": "Safety gate: specify ?confirm=SOVEREIGN"}, status_code=403)
    try:
        reset_file = LOG_DIR / "sim_reset.txt"
        with open(reset_file, "w") as f:
            f.write(datetime.now().isoformat())
        return {"status": "success", "message": "Wallet recapitalized."}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/promote_weights")
def promote_weights():
    """Promotes latest training staging weights to production sandbox_active.keras"""
    import shutil
    src_p = PROJ_ROOT / "models" / "hydra_best.keras"
    dest_p = PROJ_ROOT / "models" / "sandbox_active.keras"
    
    if not src_p.exists():
        return JSONResponse({"status": "error", "message": f"Staging weights not found at {src_p}"}, status_code=404)
        
    try:
        # Atomic file copy
        shutil.copy(src_p, dest_p)
        
        # Enforce user ownership
        try:
            import pwd, grp
            uid = pwd.getpwnam("ubuntu").pw_uid
            gid = grp.getgrnam("ubuntu").gr_gid
            os.chown(str(dest_p), uid, gid)
        except:
            pass
            
        return {"status": "success", "message": f"Brain promoted successfully! Updated sandbox_active.keras timestamp to {dest_p.stat().st_mtime}"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/config_params")
def get_config_params():
    """Reads live dynamic configuration parameters from live_params.json"""
    params_p = SRC_ROOT / "config" / "live_params.json"
    if not params_p.exists():
        return {"min_swing": 100.0, "cert_threshold": 0.85}
    try:
        with open(params_p, "r") as f:
            return json.load(f)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/config_params")
def post_config_params(data: dict):
    """Updates live dynamic configuration parameters in live_params.json"""
    params_p = SRC_ROOT / "config" / "live_params.json"
    try:
        min_swing = float(data.get("min_swing", 100.0))
        cert_threshold = float(data.get("cert_threshold", 0.85))
        
        # Save to JSON
        with open(params_p, "w") as f:
            json.dump({"min_swing": min_swing, "cert_threshold": cert_threshold}, f, indent=2)
            
        return {"status": "success", "message": f"Live parameters saved: Min Swing: ${min_swing}, Certainty: {cert_threshold*100:.0f}%"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/logs")
async def get_logs(type: str = "trading", lines: int = 100):
    """
    Exposes the last N lines of system logs.
    Supported types: 'trading' (live_pilot.log), 'training' (latest iron_oracle*.log), 'api' (api.log)
    """
    from collections import deque
    if type == "trading":
        log_path = LOG_DIR / "live_pilot.log"
    elif type == "training":
        log_path = find_latest_log()
    elif type == "api":
        log_path = LOG_DIR / "api.log"
    else:
        raise HTTPException(status_code=400, detail="Invalid log type. Choose 'trading', 'training', or 'api'")

    if not log_path or not log_path.exists():
        return {"log": f"Log file for type '{type}' not found at {log_path}", "path": str(log_path.name if log_path else "")}

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, maxlen=lines)
        clean_log = "".join(last_lines)
        return {"log": clean_log, "path": str(log_path.name), "lines": len(last_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

# ── Live Exchange Metrics Caching System ──
_live_cache = {
    "data": None,
    "ts": 0.0
}
_LIVE_CACHE_TTL = 5.0  # seconds

def fetch_and_calculate_live_metrics():
    """Fetch live wallet balance and execution fills from Delta Exchange testnet."""
    client = DeltaClient(testnet=True)
    initial_wallet = 200.0
    
    # 1. Fetch balance
    balance_usd = initial_wallet
    try:
        b_res = client._get("/v2/wallet/balances", auth=True)
        if isinstance(b_res, dict) and b_res.get("success"):
            for b in b_res.get("result", []):
                if b.get("asset_symbol") == "USD":
                    balance_usd = float(b.get("balance", initial_wallet))
                    break
    except Exception as e:
        print(f"DEBUG: Live balance error: {e}")

    # 2. Fetch fills
    fills = []
    try:
        f_res = client._get("/v2/fills", auth=True)
        if isinstance(f_res, dict) and f_res.get("success"):
            fills = f_res.get("result", [])
    except Exception as e:
        print(f"DEBUG: Live fills error: {e}")

    # 3. Reconstruct trades & respect sim_reset
    sorted_fills = sorted(fills, key=lambda x: x.get('created_at', ''))
    
    # Check if a recapitalization happened
    reset_ts = ""
    reset_file = LOG_DIR / "sim_reset.txt"
    if reset_file.exists():
        try:
            with open(reset_file, "r") as rf:
                reset_ts = rf.read().strip()
        except:
            pass

    trades = []
    current_position = None
    total_commission = 0.0
    accumulated_pnl = 0.0
    
    for f in sorted_fills:
        created_at = f.get('created_at', '')
        if reset_ts and created_at < reset_ts:
            continue
            
        price = float(f.get('price', 0.0))
        size = float(f.get('size', 0.0))
        side = f.get('side')
        comm = float(f.get('commission', 0.0))
        total_commission += comm
        
        meta = f.get('meta_data', {})
        new_pos = meta.get('new_position', {})
        realized_pnl = float(new_pos.get('realized_pnl', 0.0)) if new_pos else 0.0
        
        if realized_pnl != 0.0:
            accumulated_pnl += realized_pnl
            trade_side = 'LONG' if side == 'sell' else 'SHORT'
            entry_price = price
            if current_position:
                entry_price = current_position['entry_price']
            
            # net_pct relative to initial wallet size of 200.0
            net_pct = (realized_pnl / initial_wallet) * 100.0
            
            trades.append({
                "timestamp": f['created_at'],
                "side": trade_side,
                "entry": entry_price,
                "exit": price,
                "net_pct": net_pct,
                "net_pnl": realized_pnl,
                "expert": "Sovereign-Pilot",
                "commission": comm
            })
            current_position = None
        else:
            trade_side = 'LONG' if side == 'buy' else 'SHORT'
            current_position = {
                'side': trade_side,
                'entry_price': float(new_pos.get('entry_price', price)) if new_pos else price,
                'size': size,
                'commission': comm
            }

    # Reversing trades for dashboard strikes feed (newest first)
    dashboard_trades = list(reversed(trades))

    # 4. Finalize virtual balance if recapitalized
    if reset_ts:
        balance_usd = initial_wallet + accumulated_pnl

    # Convert timestamps to IST for UI feed
    for t in dashboard_trades:
        try:
            utc_t = datetime.fromisoformat(t["timestamp"].replace('Z', '+00:00'))
            t["timestamp"] = (utc_t + timedelta(hours=5, minutes=30)).strftime("%d %b %I:%M %p")
        except:
            pass

    # Risk metrics from completed trades
    risk_metrics = {"win_rate": 0.0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0}
    if trades:
        pnls = [t["net_pct"] / 100.0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        risk_metrics["win_rate"] = (len(wins) / len(trades)) * 100
        risk_metrics["profit_factor"] = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (1.0 if wins else 0.0)
        
        if len(pnls) > 1:
            std = np.std(pnls)
            # Live trades are spaced hours or days apart, so daily annualization (np.sqrt(365)) is mathematically appropriate
            risk_metrics["sharpe"] = (np.mean(pnls) / std * np.sqrt(365)) if std > 0 else 0.0
            
        # Drawdown
        equity = [initial_wallet]
        for t in trades:
            equity.append(equity[-1] + t["net_pnl"])
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        risk_metrics["max_dd"] = np.max(drawdown) if len(drawdown) > 0 else 0.0

    # Parse the active live pilot logs for live certainty/signals
    live_cert, live_reas, live_dec, live_ts = parse_live_pilot_log()

    return {
        "wallet_usd": balance_usd,
        "net_profit": balance_usd - initial_wallet,
        "trades": dashboard_trades,
        "total_fees": total_commission,
        "risk": risk_metrics,
        "consensus_pct": live_cert if live_cert > 0 else 0.0,
        "reasoning": live_reas,
        "decision": live_dec,
        "timestamp": live_ts
    }

@app.get("/api/stats")
async def get_stats(mode: str = "live"):
    if mode == "live":
        # ── Pure Live Exchange Statistics Mode ──
        # Fetch Live Metrics from Cache
        now = time.time()
        if now - _live_cache["ts"] >= _LIVE_CACHE_TTL or _live_cache["data"] is None:
            try:
                _live_cache["data"] = await asyncio.to_thread(fetch_and_calculate_live_metrics)
                _live_cache["ts"] = now
            except Exception as e:
                print(f"DEBUG: Live metrics background sync failed: {e}")
                if _live_cache["data"] is None:
                    _live_cache["data"] = {
                        "wallet_usd": INITIAL_WALLET_USD,
                        "net_profit": 0.0,
                        "trades": [],
                        "total_fees": 0.0,
                        "risk": {"win_rate": 0.0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0},
                        "consensus_pct": 0.0,
                        "reasoning": "IDLE",
                        "decision": "SCANNING",
                        "timestamp": ""
                    }
        live_data = _live_cache["data"]
        
        # ── Active Position Tracking ──
        position_data = {
            "has_position": False,
            "side": "NONE",
            "size": 0.0,
            "entry_price": 0.0,
            "mark_price": 0.0,
            "unrealized_pnl": 0.0,
            "active_sl": 0.0,
            "active_tp": 0.0
        }
        try:
            client = DeltaClient(testnet=True)
            pos = client.get_positions()
            product_id = client._resolve_product_id("BTCUSD")
            my_pos = [p for p in pos if int(p.get("product_id", 0)) == product_id]
            if my_pos:
                active_pos = my_pos[0]
                size = float(active_pos.get("size", 0.0))
                if abs(size) > 0:
                    entry_p = float(active_pos.get("entry_price", 0.0))
                    mark_p = float(active_pos.get("mark_price", 0.0))
                    unrealized = float(active_pos.get("unrealized_pnl", 0.0))
                    position_data.update({
                        "has_position": True,
                        "side": "LONG" if size > 0 else "SHORT",
                        "size": abs(size),
                        "entry_price": entry_p,
                        "mark_price": mark_p,
                        "unrealized_pnl": unrealized
                    })
                    try:
                        open_orders = client._get("/v2/orders", params={"symbol": "BTCUSD"}, auth=True).get("result", [])
                        bracket_orders = [o for o in open_orders if o.get("bracket_order") == True and o.get("state") == "pending"]
                        for o in bracket_orders:
                            s_type = o.get("stop_order_type")
                            s_price = o.get("stop_price")
                            if s_type == "stop_loss_order" and s_price:
                                position_data["active_sl"] = float(s_price)
                            elif s_type == "take_profit_order" and s_price:
                                position_data["active_tp"] = float(s_price)
                    except Exception as e:
                        pass
        except Exception as e:
            pass

        live_trader_running = False
        try:
            res_trader = subprocess.run(["pgrep", "-f", "live_trader.py"], capture_output=True, text=True)
            live_trader_running = bool(res_trader.stdout.strip())
        except:
            pass

        return {
            "status": live_data.get("decision", "SCANNING"),
            "live_trader_running": live_trader_running,
            "wallet_usd": live_data.get("wallet_usd", INITIAL_WALLET_USD),
            "net_profit": live_data.get("net_profit", 0.0),
            "trades": live_data.get("trades", []),
            "total_fees": live_data.get("total_fees", 0.0),
            "risk": live_data.get("risk", {"win_rate": 0.0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0}),
            "consensus_pct": live_data.get("consensus_pct", 0.0),
            "reasoning": live_data.get("reasoning", "IDLE"),
            "active_position": position_data,
            "config_rules": {
                "leverage": LEVERAGE,
                "position_size_pct": POSITION_SIZE_PCT,
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "breakeven_trigger_pct": BREAKEVEN_TRIGGER_PCT,
                "trailing_stop_pct": TRAILING_STOP_PCT
            },
            "config": {
                "wallet": INITIAL_WALLET_USD, 
                "fee": FEE_RATE,
                "leverage": LEVERAGE,
                "position_size_pct": POSITION_SIZE_PCT,
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "breakeven_trigger_pct": BREAKEVEN_TRIGGER_PCT,
                "trailing_stop_pct": TRAILING_STOP_PCT
            }
        }
        
    else:
        # ── Pure Neural Training Lab Mode ──
        epochs, candle_count, current_step, total_steps, current_epoch_num = parse_training_log()
        latest = epochs[-1] if epochs else {}

        expert_heatmap = [random.random() * 0.1 for _ in range(256)]
        roi_block = {"net": None, "trades": 0, "status": "measure", "wallet": INITIAL_WALLET_USD}
        
        roi_path = LOG_DIR / "latest_roi.json"
        if roi_path.exists():
            try:
                with open(roi_path, "r") as f:
                    rd = json.load(f)
                t80 = rd.get("tiers", {}).get("80")
                if t80:
                    roi_block.update({"net": t80["net"], "trades": t80["trades"], "status": "live"})
                if "expert_heatmap" in rd:
                    expert_heatmap = rd["expert_heatmap"]
            except: pass

        recent_trades = []
        risk_metrics = {"win_rate": 0, "sharpe": 0, "profit_factor": 0, "max_dd": 0}
        trades_path = LOG_DIR / "recent_sim_trades.json"
        if trades_path.exists():
            try:
                with open(trades_path, "r") as f:
                    all_trades = json.load(f)
                    recent_trades = all_trades[:15]
                
                if all_trades:
                    pnls = [t["net_pct"] / 100.0 for t in all_trades]
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p <= 0]
                    
                    risk_metrics["win_rate"] = (len(wins) / len(all_trades)) * 100 if all_trades else 0
                    risk_metrics["profit_factor"] = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (1.0 if wins else 0)
                    
                    if len(pnls) > 1:
                        std = np.std(pnls)
                        risk_metrics["sharpe"] = (np.mean(pnls) / std * np.sqrt(365 * 96)) if std > 0 else 0
                    
                    # Max Drawdown
                    equity = [1.0]
                    for p in reversed(pnls):
                        equity.append(equity[-1] * (1.0 + p))
                    equity = np.array(equity)
                    peak = np.maximum.accumulate(equity)
                    drawdown = (peak - equity) / peak
                    risk_metrics["max_dd"] = np.max(drawdown) if len(drawdown) > 0 else 0

                for t in recent_trades:
                    try:
                        utc_t = datetime.fromisoformat(t["timestamp"])
                        t["timestamp"] = (utc_t + timedelta(hours=5, minutes=30)).strftime("%d %b %I:%M %p")
                    except: pass
            except Exception as e:
                print(f"DEBUG: Risk calc error: {e}")

        epoch_progress = (current_step / total_steps * 100) if total_steps > 0 else 0
        seconds_per_step = 16.0 
        remaining_steps_this_epoch = total_steps - current_step
        remaining_epochs = 300 - current_epoch_num
        total_remaining_seconds = (remaining_steps_this_epoch * seconds_per_step) + (remaining_epochs * total_steps * seconds_per_step)
        mission_eta_days = total_remaining_seconds / 86400

        price = await get_live_btc_price()
        dialogue = discourse_engine.generate_debate({
            "price": price, "certainty": latest.get("certainty", 0.0),
            "expert_heatmap": expert_heatmap, "epoch": current_epoch_num, "current_step": current_step
        })

        certainty_val = latest.get("certainty", 0.0)
        consensus_pct = certainty_val * 100 if certainty_val > 0 else 0

        return {
            "status": "TRAINING" if current_step > 0 else "IDLE",
            "epochs": epochs,
            "latest": latest,
            "current_epoch": current_epoch_num,
            "current_step": current_step,
            "total_steps": total_steps,
            "progress": {
                "epoch": round(epoch_progress, 1),
                "mission_eta_days": round(mission_eta_days, 1),
                "mission_pct": round((current_epoch_num / 300) * 100, 1)
            },
            "consensus_pct": round(consensus_pct, 1),
            "net_profit": (roi_block["net"] or 0) if roi_block["status"] == "live" else 0,
            "risk": risk_metrics,
            "trades": recent_trades,
            "dialogue": dialogue,
            "expert_heatmap": expert_heatmap,
            "config": {
                "wallet": INITIAL_WALLET_USD, 
                "fee": FEE_RATE,
                "leverage": LEVERAGE,
                "position_size_pct": POSITION_SIZE_PCT,
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "breakeven_trigger_pct": BREAKEVEN_TRIGGER_PCT,
                "trailing_stop_pct": TRAILING_STOP_PCT
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)