"""
SOVEREIGN ALPHA PILOT (V11.0 — IRON ORACLE) ⚓🏛️
==================================================
- Timeframe: 15 minutes (Phase 5 — Maximum SNR)
- Brain: IRON ORACLE V11.0 (256-Expert MoE + MLA + RoPE)
- Targets: MTP-15 (Price, Volatility, Volume Flow)
- Exchange: Delta Exchange (Market-Order + SL/TP Brackets)
- Filter: Certainty threshold (80%+ = Sovereign Edge)
"""

import os, sys, time
import numpy as np
import keras
import pandas as pd
from datetime import datetime
from pathlib import Path

# ROOT must point to the project root (/kat/), not the file's own directory
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.hydra import (HydraBlock, GatedMoE, MLALayer,
                        RMSNorm, TurboQuant, SwiGLU,
                        SovereignLoss, CertaintyMetric, SovereignAccuracy)
from exchange.delta_client import DeltaClient
from exchange.fetch_data   import fetch_live_kat_data
from data.preprocess       import build_feature_cols, compute_indicators

# ── CONFIG ────────────────────────────────────────────────────────────────────
from config.sovereign_config import (LEVERAGE, POSITION_SIZE_PCT, MAX_TRADES_PER_DAY,
                                     BREAKEVEN_TRIGGER_PCT, TRAILING_STOP_PCT)
SYMBOL         = "BTCUSD"
SIZE           = 1              # Legacy fallback contract size
MIN_SWING      = 100.0          # Only trade if expected move > $100 (to beat fees)
THRESHOLD      = 0.08           # Increased base conviction (was 0.05)
TIMEFRAME      = "15m"          # Match training timeframe (15m)
CTX_WIN        = 120            # Context window (30 hours)
SLEEP_S        = 900            # 15 minute polling
CERT_THRESHOLD = 0.85           # SNIPER MODE: Only 85%+ certainty
COOLDOWN_BARS  = 1              # Prevent back-to-back flipping (saves fees)

# ── HUD ───────────────────────────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREEN  = "\033[32m"
C_RED    = "\033[31m"
C_CYAN   = "\033[36m"
C_YELLOW = "\033[33m"

def log(msg, color=C_RESET):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{C_CYAN}{stamp}{C_RESET}] {color}{msg}{C_RESET}", flush=True)

def load_model():
    """Load the trained Iron Oracle brain from the isolated sandbox copy."""
    from core.hydra import build_kraken
    from data.preprocess import build_feature_cols
    model_p = ROOT / "models" / "sandbox_active.keras"
    if not model_p.exists():
        log(f"❌ No sandbox model found at {model_p}", C_RED); sys.exit(1)
    n_feat = len(build_feature_cols())
    log(f"🏗️  Re-building Iron Oracle V12.0 ({n_feat} features) | Loading sandbox active brain...")
    model = build_kraken(n_features=n_feat, context_window=CTX_WIN)
    model.load_weights(str(model_p))
    mtime = model_p.stat().st_mtime
    log(f"✅ Sandbox Brain sync complete: sandbox_active.keras (mtime: {mtime})", C_GREEN)
    return model, mtime

def get_neural_signal(model):
    """
    Fetch latest 15m market data, engineer features, run inference.
    FIX #1: Uses Dynamic Local Scaling (DLS) — matching training pipeline.
    Returns (mean_price_move, certainty_pct, mean_volatility, full_pred_array).
    """
    try:
        features = build_feature_cols()
        n_feats  = len(features)  # 42

        # Fetch CTX_WIN + buffer for indicator warm-up
        df = fetch_live_kat_data(symbol=SYMBOL, n_candles=CTX_WIN + 150, timeframe=TIMEFRAME)
        if df is None or len(df) == 0:
            raise ValueError("fetch_live_kat_data returned empty or None dataframe")
            
        df_feat = compute_indicators(df)
        data    = df_feat[features].values.astype("float32")

        # FIX #1: DLS — scale using the local window stats (same as training)
        x_raw  = data[-CTX_WIN:]
        l_mean = x_raw.mean(axis=0)
        l_std  = x_raw.std(axis=0) + 1e-8
        x_scaled = (x_raw - l_mean) / l_std
        X_in   = x_scaled[np.newaxis].astype("float32")  # (1, 120, 42)

        # Iron Oracle returns [prediction_trajectory, certainty_map, reasoning_head]
        outputs      = model(X_in, training=False)
        pred         = outputs[0].numpy()[0]   # (16, 3)
        certainty_2d = outputs[1].numpy()[0]   # (120,) per-step certainty
        reasoning    = int(np.argmax(outputs[2].numpy()[0]))

        pred_future  = pred[1:]                # (15, 3) — future steps only
        p_anchor     = pred[0, 0]              # Z-score at anchor price
        p_curve      = pred_future[:, 0]       # price trajectory
        v_curve      = pred_future[:, 1]       # volatility

        p_change     = p_curve - p_anchor      # Trajectory delta relative to anchor

        # Normalize certainty to 0–100%
        cert_mean    = float(np.mean(certainty_2d))
        cert_pct     = cert_mean

        t_close      = features.index('close')
        close_std    = l_std[t_close]

        return np.mean(p_change), cert_pct, np.mean(v_curve), reasoning, pred_future, close_std
    except Exception as e:
        import traceback
        log(f"❌ Error in get_neural_signal: {e}", C_RED)
        log(traceback.format_exc(), C_RED)
        raise e

def run_pilot():
    from config.sovereign_config import LABELS
    print("="*60)
    print(f"  ⚓ IRON ORACLE V11.0 — [ {SYMBOL} ] LIVE PILOT")
    print(f"  📡 Timeframe : {TIMEFRAME}")
    print(f"  🎯 Certainty : {CERT_THRESHOLD*100:.0f}%+ required to trade")
    print(f"  💰 Position  : {SIZE} contract(s) | {LEVERAGE}x Leverage")
    print("="*60)

    client = DeltaClient(testnet=True)
    
    # Programmatically set leverage to configured value
    log(f"⚙️  Enforcing {LEVERAGE}x Leverage on Exchange for {SYMBOL}...", C_CYAN)
    try:
        lev_resp = client.set_leverage(SYMBOL, LEVERAGE)
        if lev_resp and lev_resp.get("success", False):
            log(f"✅ Exchange Leverage successfully locked at {LEVERAGE}x!", C_GREEN)
        else:
            log(f"⚠️  Leverage sync warning: {lev_resp}", C_YELLOW)
    except Exception as e:
        log(f"⚠️  Could not programmatically set leverage: {e}", C_YELLOW)

    model, last_mtime = load_model()  # FIX #1: DLS — no scaler needed

    last_trade_time = 0
    daily_trades = []  # Keep track of rolling 24-hour trade timestamps
    peak_price = 0.0
    breakeven_activated = False
    exchange_sl_price = 0.0  # Track the current active SL price registered on the exchange
    original_tp_price = 0.0  # Track the original Take Profit price from exchange
    while True:
        try:
            # ── Hot-Reload Check (Sandbox Active) ────────────────────────────
            model_p = ROOT / "models" / "sandbox_active.keras"
            if model_p.exists():
                curr_mtime = model_p.stat().st_mtime
                if curr_mtime > last_mtime:
                    log(f"🔄 ACTIVE SANDBOX BRAIN UPDATE DETECTED! Hot-reloading weights...", C_CYAN)
                    model.load_weights(str(model_p))
                    last_mtime = curr_mtime
                    log("✅ Hot-reload complete. Active model upgraded in real-time!", C_GREEN)

            current_time = time.time()

            # ── Active Position Monitoring (Trailing Stop & Breakeven) ──────
            pos = client.get_positions()
            product_id = client._resolve_product_id(SYMBOL)
            my_pos = [p for p in pos if int(p["product_id"]) == product_id]
            is_in_long = any(float(p["size"]) > 0 for p in my_pos)
            is_in_short = any(float(p["size"]) < 0 for p in my_pos)
            has_active_position = (is_in_long or is_in_short)
            
            if has_active_position:
                try:
                    active_pos = my_pos[0]
                    size = float(active_pos.get('size', 0.0))
                    entry_p = float(active_pos.get('entry_price', 0.0))
                    mark_p = float(active_pos.get('mark_price', 0.0))
                    
                    is_long = size > 0
                    
                    # 1. Fetch active brackets directly from Delta Exchange to update state variables
                    try:
                        open_orders = client._get("/v2/orders", params={"symbol": SYMBOL}, auth=True).get("result", [])
                        bracket_orders = [o for o in open_orders if o.get("bracket_order") == True and o.get("state") == "pending"]
                        
                        exchange_sl_price = 0.0
                        original_tp_price = 0.0
                        for o in bracket_orders:
                            s_type = o.get("stop_order_type")
                            s_price = o.get("stop_price")
                            if s_type == "stop_loss_order" and s_price:
                                exchange_sl_price = float(s_price)
                            elif s_type == "take_profit_order" and s_price:
                                original_tp_price = float(s_price)
                    except Exception as e:
                        log(f"⚠️ Failed to sync active exchange brackets: {e}", C_YELLOW)
                    
                    # Estimate if not set on exchange yet
                    if exchange_sl_price == 0.0:
                        exchange_sl_price = entry_p * (1 - 0.012) if is_long else entry_p * (1 + 0.012)
                    
                    # Update peak price
                    if is_long:
                        if peak_price == 0.0 or mark_p > peak_price:
                            peak_price = mark_p
                        profit_pct = (mark_p - entry_p) / entry_p * 100
                    else:
                        if peak_price == 0.0 or mark_p < peak_price:
                            peak_price = mark_p
                        profit_pct = (entry_p - mark_p) / entry_p * 100
                        
                    # Breakeven Activation & Exchange SL Update
                    if profit_pct >= BREAKEVEN_TRIGGER_PCT and not breakeven_activated:
                        # Double check if exchange-side SL is already locked at breakeven
                        is_sl_at_breakeven = False
                        if is_long and exchange_sl_price >= entry_p:
                            is_sl_at_breakeven = True
                        elif not is_long and exchange_sl_price <= entry_p:
                            is_sl_at_breakeven = True
                            
                        if not is_sl_at_breakeven:
                            breakeven_activated = True
                            log(f"🛡️  BREAKEVEN TRIGGERED: Profit hit +{profit_pct:.2f}% (Price: ${mark_p:.2f} | Entry: ${entry_p:.2f}). Stop Loss is now locking at break-even on Delta Exchange.", C_GREEN)
                            try:
                                sl_price = round(entry_p, 1)
                                client.update_bracket_order(SYMBOL, sl_price=sl_price)
                                exchange_sl_price = sl_price
                                log(f"✅ DELTA EXCHANGE SERVER SL UPDATED: Locked Stop Loss at entry ${sl_price} on Exchange UI!", C_GREEN)
                            except Exception as api_err:
                                log(f"⚠️ Failed to update breakeven SL on Delta Server: {api_err}", C_YELLOW)
                                breakeven_activated = False # retry next loop if failed
                        else:
                            breakeven_activated = True
                        
                    # Breakeven Floor Enforcement (Python Fallback)
                    if breakeven_activated:
                        if is_long and mark_p <= entry_p:
                            log(f"🚨 BREAKEVEN FLOOR HIT: Price ${mark_p:.2f} returned to entry ${entry_p:.2f}. Closing position to protect capital.", C_YELLOW)
                            client.place_order(SYMBOL, abs(size), "sell" if is_long else "buy")
                            peak_price = 0.0
                            breakeven_activated = False
                            exchange_sl_price = 0.0
                            time.sleep(10); continue
                        elif not is_long and mark_p >= entry_p:
                            log(f"🚨 BREAKEVEN FLOOR HIT: Price ${mark_p:.2f} returned to entry ${entry_p:.2f}. Closing position to protect capital.", C_YELLOW)
                            client.place_order(SYMBOL, abs(size), "sell" if is_long else "buy")
                            peak_price = 0.0
                            breakeven_activated = False
                            exchange_sl_price = 0.0
                            time.sleep(10); continue
                            
                    # Trailing Stop Enforcement
                    if profit_pct >= 0.5:
                        target_ts_price = round(peak_price * (1 - TRAILING_STOP_PCT/100) if is_long else peak_price * (1 + TRAILING_STOP_PCT/100), 1)
                        # Institutional Trailing Take Profit: Keeps moving target 2.8% ahead of the peak to let profits run
                        target_tp_price = round(peak_price * (1 + 2.8/100) if is_long else peak_price * (1 - 2.8/100), 1)
                        
                        # Python Fallback: If price pulls back below target stop, close immediately!
                        if (is_long and mark_p <= target_ts_price) or (not is_long and mark_p >= target_ts_price):
                            log(f"🚨 TRAILING STOP TRIGGERED: Profit pulled back below target ${target_ts_price} (Current: ${mark_p:.2f}). Locking in profits!", C_GREEN)
                            client.place_order(SYMBOL, abs(size), "sell" if is_long else "buy")
                            peak_price = 0.0
                            breakeven_activated = False
                            exchange_sl_price = 0.0
                            time.sleep(10); continue
                            
                        # Exchange-Side Trailing Update: If target stop has moved up significantly (>= $50), update exchange!
                        should_update_exchange = False
                        if is_long and target_ts_price > (exchange_sl_price + 50.0):
                            should_update_exchange = True
                        elif not is_long and target_ts_price < (exchange_sl_price - 50.0):
                            should_update_exchange = True
                            
                        if should_update_exchange:
                            try:
                                log(f"📈 TRAILING BRACKET UPDATING: Trailing SL to ${target_ts_price} and Trailing TP to ${target_tp_price} on Delta Exchange.", C_CYAN)
                                client.update_bracket_order(SYMBOL, sl_price=target_ts_price, tp_price=target_tp_price)
                                exchange_sl_price = target_ts_price
                                original_tp_price = target_tp_price
                                log(f"📈 DELTA EXCHANGE SERVER BRACKET DYNAMICALLY TRAILED: Adjusted Stop Loss to ${target_ts_price} and Take Profit to ${target_tp_price} on Exchange UI!", C_GREEN)
                            except Exception as api_err:
                                log(f"⚠️ Failed to update trailing bracket on Delta Server: {api_err}", C_YELLOW)
                        
                    # Status Log
                    side_str = "LONG" if is_long else "SHORT"
                    log(f"⚡ MONITORING: {side_str} ({abs(size)} contracts) | Entry: ${entry_p:.2f} | Mark: ${mark_p:.2f} | P/L: +{profit_pct:.2f}% | Peak: ${peak_price:.2f} | BE: {breakeven_activated} | Active Exchange SL: ${exchange_sl_price:.2f} | Preserved TP: ${original_tp_price:.2f}", C_CYAN)
                except Exception as monitor_err:
                    log(f"⚠️ Error in active position monitor: {monitor_err}", C_YELLOW)
                    
                # High-speed adaptive sleep (10s)
                time.sleep(10); continue
            else:
                peak_price = 0.0
                breakeven_activated = False
                exchange_sl_price = 0.0

            # ── Rolling 24h Daily Circuit Breaker ────────────────────────────
            daily_trades = [t for t in daily_trades if current_time - t < 86400]
            if len(daily_trades) >= MAX_TRADES_PER_DAY:
                log(f"🛑 DAILY CIRCUIT BREAKER ACTIVE ({len(daily_trades)}/{MAX_TRADES_PER_DAY} trades taken in last 24h). Halting new entries.", C_RED)
                time.sleep(SLEEP_S); continue

            log(f"📡 Polling {SYMBOL} [{TIMEFRAME}] market stream...")

            # ── Inference ────────────────────────────────────────────────────
            mean_price, cert_raw, mean_vol, reasoning, pred, close_std = get_neural_signal(model)

            # Get current price for swing calculation
            df_curr = fetch_live_kat_data(SYMBOL, 1, TIMEFRAME)
            curr_price = df_curr['close'].iloc[-1]
            
            # Predict actual dollar move (using actual local standard deviation)
            est_swing = abs(mean_price) * close_std

            cert_norm = cert_raw

            log(f"🔮 REASONING     : {LABELS[reasoning]}")
            log(f"🔮 EXPECTED SWING: ±${est_swing:.2f}  (min: ${MIN_SWING:.0f})")
            log(f"🧠 CERTAINTY     : {cert_norm*100:.1f}%  (threshold: {CERT_THRESHOLD*100:.0f}%)")

            # ── Sniper Gate 1: Certainty ─────────────────────────────────────
            if cert_norm < CERT_THRESHOLD:
                log(f"🔕 CERTAINTY TOO LOW ({cert_norm*100:.1f}%) — HOLDING", C_YELLOW)
                time.sleep(SLEEP_S); continue

            # ── Sniper Gate 2: Sovereign Reasoning (Fee Awareness) ───────────
            if reasoning not in [0, 1]:
                log(f"🛑 REASONING: {LABELS[reasoning]} — Potential fee loss. HOLDING", C_YELLOW)
                time.sleep(SLEEP_S); continue

            # ── Sniper Gate 3: Fee Protection (Swing Size) ────────────────────
            if est_swing < MIN_SWING:
                log(f"📉 SWING TOO SMALL (${est_swing:.2f} < ${MIN_SWING:.0f}) — FEES WOULD EAT PROFIT", C_YELLOW)
                time.sleep(SLEEP_S); continue

            # ── Sniper Gate 3: Cooldown ──────────────────────────────────────
            if (current_time - last_trade_time) < (COOLDOWN_BARS * SLEEP_S):
                log(f"⏳ COOLDOWN ACTIVE — Saving fees.", C_YELLOW)
                time.sleep(SLEEP_S); continue

            # ── Dynamic Armor (Mastery Enhancement) ──────────────────────────
            # Armor widens during high volatility to prevent "Stop Loss Hunting"
            # and narrows during low volatility to lock in profits.
            base_sl = 1.2
            base_tp = 2.8
            vol_multiplier = 1.0 + min(0.5, abs(mean_vol)) # Max +50% widening
            
            dyn_sl = base_sl * vol_multiplier
            dyn_tp = base_tp * vol_multiplier

            # ── Dynamic Sizing Strategy ───────────────────────────────────────
            dynamic_size = SIZE
            try:
                balances = client._get('/v2/wallet/balances', auth=True).get('result', [])
                usd_bal = [b for b in balances if b.get('asset_symbol') == 'USD']
                if usd_bal:
                    available_usd = float(usd_bal[0].get('available_balance', 0.0))
                    allowed_margin = available_usd * POSITION_SIZE_PCT
                    allowed_notional_usd = allowed_margin * LEVERAGE
                    
                    # 1 contract value = 0.001 * current BTCUSD price
                    ob = client.get_orderbook(SYMBOL)
                    best_bid = float(ob['buy'][0]['price'])
                    best_ask = float(ob['sell'][0]['price'])
                    mid_p   = (best_bid + best_ask) / 2
                    contract_val_usd = 0.001 * mid_p
                    
                    dynamic_size = max(1, int(allowed_notional_usd / contract_val_usd))
                    log(f"💰 POSITION SIZING: Available USD: ${available_usd:.2f} | Target Margin: ${allowed_margin:.2f} | Dynamic Size: {dynamic_size} contracts", C_CYAN)
            except Exception as size_err:
                log(f"⚠️ Sizing calculation error: {size_err}. Falling back to default SIZE: {SIZE}", C_YELLOW)

            # ── Signal Logic ──────────────────────────────────────────────────
            dynamic_thresh = THRESHOLD * (1.0 + max(0.0, mean_vol))
            
            pos        = client.get_positions()
            product_id = client._resolve_product_id(SYMBOL)
            my_pos     = [p for p in pos if int(p["product_id"]) == product_id]
            is_in_long  = any(float(p["size"]) > 0 for p in my_pos)
            is_in_short = any(float(p["size"]) < 0 for p in my_pos)

            if mean_price > dynamic_thresh:
                if not is_in_long:
                    log(f"📈 LONG signal (+${est_swing:.2f}) @ {cert_norm*100:.1f}%", C_GREEN)
                    log(f"🛡️  DYNAMIC ARMOR: SL {dyn_sl:.2f}% | TP {dyn_tp:.2f}%")
                    client.place_order(SYMBOL, dynamic_size, "buy", sl_pct=dyn_sl, tp_pct=dyn_tp)
                    last_trade_time = time.time()
                    daily_trades.append(last_trade_time)
                else: log(f"✅ Already LONG", C_GREEN)

            elif mean_price < -dynamic_thresh:
                if not is_in_short:
                    log(f"📉 SHORT signal (-${est_swing:.2f}) @ {cert_norm*100:.1f}%", C_RED)
                    log(f"🛡️  DYNAMIC ARMOR: SL {dyn_sl:.2f}% | TP {dyn_tp:.2f}%")
                    client.place_order(SYMBOL, dynamic_size, "sell", sl_pct=dyn_sl, tp_pct=dyn_tp)
                    last_trade_time = time.time()
                    daily_trades.append(last_trade_time)
                else: log(f"✅ Already SHORT", C_RED)
            else:
                log(f"💤 No signal (Neutral Zone) — Waiting.", C_RESET)
            log(f"💤 Next poll in {SLEEP_S}s...")
            time.sleep(SLEEP_S)
        except KeyboardInterrupt:
            log("🛑 Pilot stopped by operator.", C_YELLOW)
            break
        except Exception as e:
            import traceback
            log(f"⚠️  Loop error: {e}", C_RED)
            log(traceback.format_exc(), C_RED)
            time.sleep(30)

if __name__ == "__main__":
    run_pilot()
