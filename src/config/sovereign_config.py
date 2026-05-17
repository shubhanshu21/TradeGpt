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
POSITION_SIZE_PCT  = 1.0    # 1.0 = 100% of current wallet (Full-Port)
PROFIT_GOAL_PCT    = 0.1    # 10% profit target for UI progress bars
RISK_MULTIPLIER    = 100.0  # Max Drawdown penalty for Station Health   # 1.0 = 100% (Full-Port) | 0.1 = 10%
SOVEREIGN_MULTIPLIER = 1.0  # Strategic Scaling Constant

# Neural Architecture
CONTEXT_WINDOW = 120
FORECAST_STEPS = 15

# Reasoning head label map
LABELS = {0: "SOVEREIGN_LONG 🏹", 1: "SOVEREIGN_SHORT 📉", 2: "FEE_TRAP ⚠️", 3: "NOISE 😴"}
