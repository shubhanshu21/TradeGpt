import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exchange.delta_client import DeltaClient

client = DeltaClient(testnet=True)
try:
    print("Querying /v2/fills...")
    fills = client._get("/v2/fills", auth=True)
    print("Fills keys:", list(fills.keys()) if isinstance(fills, dict) else "Not a dict")
    print("Fills result:", fills.get("result", [])[:3] if isinstance(fills, dict) else fills)
except Exception as e:
    print("Fills failed:", e)

try:
    print("\nQuerying /v2/orders (state=filled)...")
    orders = client._get("/v2/orders", params={"state": "filled"}, auth=True)
    print("Orders keys:", list(orders.keys()) if isinstance(orders, dict) else "Not a dict")
    print("Orders result:", orders.get("result", [])[:3] if isinstance(orders, dict) else orders)
except Exception as e:
    print("Orders failed:", e)
