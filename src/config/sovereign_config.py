"""
SOVEREIGN KRAKEN CONFIGURATION (V11.2) ⚓🏛️🧠
==========================================
Central authority for strategic parameters and exchange standards.
"""

# Delta Exchange India: 0.05% Taker + 18% GST ≈ 0.059% per side
# Rounding to 0.12% for full trade (entry + exit) + slippage safety.
FEE_RATE = 0.0012
CURRENT_FEE_PCT = FEE_RATE  # Legacy Alias

# Position Management
DEFAULT_POS_SIZE_USD = 200.0
SOVEREIGN_MULTIPLIER = 1.0  # Strategic Scaling Constant

# Neural Architecture
CONTEXT_WINDOW = 120
FORECAST_STEPS = 15
