"""
SOVEREIGN KRAKEN CONFIGURATION — Equity Swing Edition
==========================================
Central authority for strategic parameters and market standards.

Converted from a crypto perpetual-futures system to Indian equity delivery
(CNC) swing trading — same core ML pipeline (transformer + MoE, prediction/
certainty/reasoning heads), different market: NSE cash equity, daily candles,
multi-day holds, real Zerodha delivery cost structure instead of crypto
perpetual fees. The broker integration, backtest engine, and cost model were
originally validated in a separate project (15-strategy bull/correction
regime testing confirmed this cost model and swing structure — 5% stop /
10% target / 20-day max hold — are realistic, not just theoretical) and are
now fully ported into this project (see config/settings.yaml,
src/backtest/delivery_costs.py, src/exchange/brokers/) — this is a
self-contained system, not dependent on any external project.
"""

import yaml
from pathlib import Path

_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_settings() -> dict:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


_settings = _load_settings()

# ── Real Zerodha DELIVERY (CNC) round-trip cost, representative trade ────────
# Delivery trades: zero brokerage, but STT on BOTH legs (not just sell), a
# flat per-sell DP charge that does NOT scale down with position size (hurts
# small trades disproportionately), and 18% GST on top of the fee components.
# For a representative ~Rs 20,000 position this computes to ~0.32% round-trip
# — notably higher than crypto's ~0.12%, driven mostly by the flat DP charge.
# This constant is an approximation for training-label gating only; the real,
# authoritative cost model (varies with exact position size) is
# src/backtest/delivery_costs.py, driven by config/settings.yaml -> swing.costs
# — that's what actually governs backtest/paper/live P&L, not this constant.
FEE_RATE = 0.0032
CURRENT_FEE_PCT = FEE_RATE  # Legacy alias, kept for code that still imports this name

# Real Zerodha delivery cost components — read from config/settings.yaml so
# there's one source of truth shared with src/backtest/delivery_costs.py,
# not a second hardcoded copy that could silently drift out of sync.
_costs = _settings["swing"]["costs"]
STT_PCT            = _costs["stt_pct"]
EXCHANGE_TXN_PCT    = _costs["exchange_txn_pct"]
SEBI_PCT            = _costs["sebi_pct"]
STAMP_DUTY_BUY_PCT  = _costs["stamp_duty_buy_pct"]
DP_CHARGE_FLAT      = _costs["dp_charge_flat"]
GST_PCT             = _costs["gst_pct"]

# Sovereign Wallet Management — read from config/settings.yaml -> swing: block
INITIAL_WALLET_INR = float(_settings["swing"]["initial_capital"])
CAPITAL_PER_TRADE  = float(_settings["swing"]["capital_per_trade"])
MAX_CONCURRENT_POSITIONS = _settings["swing"]["max_concurrent_positions"]
MAX_TRADES_PER_DAY = 4          # Rolling 24h circuit breaker (ML-pipeline-specific, not in settings.yaml)
PROFIT_GOAL_PCT    = 0.1        # 10% profit target for UI progress bars
RISK_MULTIPLIER    = 100.0      # UI display divisor for station health bar

# Neural Architecture — defaults for train.py when --context_window/--forecast_steps
# aren't passed explicitly on the CLI. Context window is in TRADING DAYS now,
# not hours — 60 trading days (~3 months) of lookback informing a swing entry
# decision that gets held for up to MAX_HOLDING_DAYS.
CONTEXT_WINDOW = 60
FORECAST_STEPS = MAX_HOLDING_DAYS = _settings["swing"]["max_holding_days"]
CERT_THRESHOLD = 0.85

# Swing trade definition — read from config/settings.yaml -> swing: block
# exactly, since the model's training labels need to predict the SAME outcome
# the real backtest/live engine actually trades on, not an approximated one.
HARD_STOP_LOSS_PCT = _settings["swing"]["hard_stop_loss_pct"]
HARD_TARGET_PCT    = _settings["swing"]["hard_target_pct"]

# Reasoning head label map
LABELS = {0: "SOVEREIGN_LONG 🏹", 1: "SOVEREIGN_SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}

# Universe — read from config/settings.yaml -> universe.symbols, so training
# and live trading can never silently use different universes.
UNIVERSE_SYMBOLS = _settings["universe"]["symbols"]

# Training can pool a smaller subset than the full tradeable universe (CPU
# training of everything at once is slow) - config/settings.yaml ->
# training.symbols, falling back to the full universe if null/omitted.
TRAINING_SYMBOLS = _settings.get("training", {}).get("symbols") or UNIVERSE_SYMBOLS
