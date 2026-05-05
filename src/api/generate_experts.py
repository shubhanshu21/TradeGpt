import json
import random
import os

expert_roles = [
    "Momentum Scalper", "Volatility Arbitrageur", "Risk Management Auditor",
    "Liquidity Provisioner", "Order Flow Analyst", "Macro Sentiment Tracker",
    "Delta-Neutral Hedger", "Trend Reversal Scout", "High-Frequency Striker",
    "Mean Reversion Specialist", "Gated Neural Guardian", "Certainty Threshold Sentry",
    "Execution Optimization Node", "Slippage Mitigation Lead", "Funding Rate Analyst",
    "Gamma-Exposure Monitor", "Cross-Exchange Arbiter", "Deep Feature Interpreter"
]

expert_personalities = [
    "Aggressive", "Conservative", "Skeptical", "Analytical", "Cautious", 
    "Confident", "Observational", "Paranoid", "Mathematical", "Intuitive"
]

def generate_experts():
    experts = {}
    for i in range(256):
        role = random.choice(expert_roles)
        personality = random.choice(expert_personalities)
        bias = random.uniform(-1, 1) # -1: Bearish, 1: Bullish
        experts[str(i)] = {
            "id": i,
            "name": f"{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Neon', 'Cipher', 'Vortex', 'Neural', 'Flow', 'Volt', 'Zenith', 'Apex'])}-{role.split()[0]} #{i:03d}",
            "role": role,
            "personality": personality,
            "bias": round(bias, 2)
        }
    return experts

if __name__ == "__main__":
    experts = generate_experts()
    out_path = "/var/www/html/ML/kat/src/api/experts_pool.json"
    with open(out_path, "w") as f:
        json.dump(experts, f, indent=4)
    print(f"Generated 256 Expert Personas in {out_path}")
