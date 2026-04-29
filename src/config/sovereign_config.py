"""
SOVEREIGN ALPHA — Strategic Configuration ⚓🏛️
==============================================
Centralized control for fees, risk thresholds, and "Sovereign Gates."
Changing values here will update the Data Engine (Preprocess) and the Trader.
"""

# ── EXCHANGE FEES ─────────────────────────────────────────────────────────────
# Default for Delta Exchange India is ~0.06% (round trip)
# Update this if the exchange changes their fee structure.
CURRENT_FEE_PCT = 0.0006  

# ── SOVEREIGN GATES ───────────────────────────────────────────────────────────
# How much profit (after fees) do we need to see to consider it a "Win"?
# 1.0 = Breakeven (Profit = Fee)
# 2.0 = Conservative (Profit must be 2x the Fee)
# 1.5 = Balanced (Model is more aggressive)
SOVEREIGN_MULTIPLIER = 2.0

# ── DYNAMIC THRESHOLD CALCULATION ────────────────────────────────────────────
# The Data Engine uses this to calculate the Z-Score (Standard Deviation) 
# required to be "Sovereign." 
def get_sovereign_threshold(volatility_avg=0.002):
    """
    Translates the % Fee into a Z-Score threshold.
    Higher volatility = tighter threshold (need bigger moves).
    """
    target_move = CURRENT_FEE_PCT * SOVEREIGN_MULTIPLIER
    # Approximate Z-Score needed: (Target % Move / Avg Volatility)
    return max(0.1, target_move / (volatility_avg + 1e-9))

# ── REASONING LABELS ──────────────────────────────────────────────────────────
# Mapping for the model's Reasoning Head
LABELS = {
    0: "SOVEREIGN_LONG",   # Net Profit > Fee * Multiplier
    1: "SOVEREIGN_SHORT",  # Net Profit > Fee * Multiplier
    2: "FEE_TRAP",         # Direction correct, but not enough to cover fees
    3: "NOISE"             # Sideways / Toxic market
}
