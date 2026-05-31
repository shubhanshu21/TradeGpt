"""
SOVEREIGN KRAKEN CONFIGURATION (V11.2) ⚓🏛️🧠
==========================================
Central authority for strategic parameters and exchange standards.
"""

# Delta Exchange India: 0.05% Taker + 18% GST ≈ 0.059% per side
# Rounding to 0.12% for full trade (entry + exit) + slippage safety.
FEE_RATE = 0.0012
CURRENT_FEE_PCT = FEE_RATE  # Legacy Alias

# Sovereign Wallet Management
INITIAL_WALLET_USD = 200.0  # Mission Starting Capital
POSITION_SIZE_PCT  = 0.25    # Fraction of available wallet USD to use as margin per trade (0.25 = 25%)
LEVERAGE           = 10     # Dynamic order leverage (10 = 10x)
MAX_TRADES_PER_DAY = 4      # Rolling 24h circuit breaker (max allowed entries to limit daily drawdown)
PROFIT_GOAL_PCT    = 0.1    # 10% profit target for UI progress bars
RISK_MULTIPLIER    = 100.0  # Max Drawdown penalty for Station Health   # 1.0 = 100% (Full-Port) | 0.1 = 10%
SOVEREIGN_MULTIPLIER = 1.0  # Strategic Scaling Constant

# Neural Architecture
CONTEXT_WINDOW = 120
FORECAST_STEPS = 15
CERT_THRESHOLD = 0.85

# Reasoning head label map
LABELS = {0: "SOVEREIGN_LONG 🏹", 1: "SOVEREIGN_SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}

# Trailing Stop & Profit Protection
BREAKEVEN_TRIGGER_PCT = 0.8   # Risk-free trigger (moves SL to entry once in 0.8% profit)
TRAILING_STOP_PCT     = 0.5   # Trail Stop Loss 0.5% behind the highest price reached

