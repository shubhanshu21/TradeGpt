"""
⚡ SOVEREIGN KRAKEN — Test Trade Executor ⚓💸
==============================================
Places a single market buy order of 1 contract of BTCUSD on the Delta India Testnet,
queries the live position, and prints out the exact leverage and margin details.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exchange.delta_client import DeltaClient
from config.sovereign_config import LEVERAGE

def test_execution():
    print("=" * 60)
    print("🚀 SOVEREIGN EXECUTION TEST — PLACING 1 CONTRACT")
    print("=" * 60)

    client = DeltaClient(testnet=True)
    
    # 1. Fetch current positions before order
    print("🔍 Fetching existing positions...")
    positions = client.get_positions()
    prod_id = client._resolve_product_id("BTCUSD")
    
    # Close any existing BTCUSD position first for a clean test
    for p in positions:
        if int(p["product_id"]) == prod_id and float(p["size"]) != 0:
            print(f"🧹 Closing existing BTCUSD position of size {p['size']}...")
            client.place_order(
                symbol="BTCUSD",
                size=abs(float(p["size"])),
                side="sell" if float(p["size"]) > 0 else "buy",
                order_type="market_order"
            )
            time.sleep(2)
    
    # 1.5. Configure Leverage to global config value
    print(f"\n⚙️ Configuring Order Leverage to {LEVERAGE}x on Exchange...")
    lev_resp = client.set_leverage("BTCUSD", LEVERAGE)
    if lev_resp and lev_resp.get("success", False):
        print(f"✅ Leverage set to {LEVERAGE}x successfully!")
    else:
        print(f"⚠️ Failed to set leverage: {lev_resp}")
    
    # 2. Place exactly 15 BUY market order with standard 1.2% SL and 2.8% TP
    print("\n🛒 Placing market BUY order of 15 contracts with 1.2% SL & 2.8% TP...")
    resp = client.place_order(
        symbol="BTCUSD",
        size=15,
        side="buy",
        order_type="market_order",
        sl_pct=1.2,
        tp_pct=2.8
    )
    
    if resp and resp.get("success", False):
        print("✅ Order placed successfully!")
    else:
        print(f"❌ Order placement failed: {resp}")
        return

    # 3. Wait for order to register on exchange
    print("\n⏳ Awaiting position confirmation (3s)...")
    time.sleep(3)

    # 4. Fetch the newly opened position and print detailed metrics
    positions = client.get_positions()
    active_pos = None
    for p in positions:
        if int(p["product_id"]) == prod_id and float(p["size"]) != 0:
            active_pos = p
            break

    if active_pos:
        print("=" * 60)
        print("📊 LIVE POSITION METRICS")
        print("=" * 60)
        print(f"   Symbol             : BTCUSD (ID: {active_pos.get('product_id')})")
        print(f"   Size               : {active_pos.get('size')} Contract(s)")
        print(f"   Entry Price        : ${float(active_pos.get('entry_price', 0.0)):,.2f}")
        print(f"   Mark Price         : ${float(active_pos.get('mark_price', 0.0)):,.2f}")
        
        # Calculate leverage and margin
        leverage = active_pos.get("leverage", "N/A")
        margin = active_pos.get("margin", "N/A")
        liq_price = active_pos.get("liquidation_price", "N/A")
        
        print(f"   Active Leverage    : {leverage}x")
        print(f"   Margin Allocated   : ${float(margin):,.4f} USD")
        print(f"   Liquidation Price  : ${float(liq_price):,.2f}" if liq_price != "N/A" else "   Liquidation Price  : N/A")
        print(f"   Unrealized PnL     : ${float(active_pos.get('unrealized_pnl', 0.0)):,.4f} USD")
        print("=" * 60)
    else:
        print("❌ Could not retrieve active position. Position might have closed or order did not execute.")

if __name__ == "__main__":
    test_execution()
