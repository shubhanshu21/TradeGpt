import time
import hashlib
import hmac
import requests
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class DeltaClient:
    """
    Simplified REST client for Delta Exchange India.
    """
    def __init__(self, testnet: bool = False):
        self.base_url = (
            "https://cdn-ind.testnet.deltaex.org"
            if testnet else
            "https://api.india.delta.exchange"
        )
        self.api_key    = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "KAT-Neural-Trader/1.0"
        })

    def _generate_signature(self, method: str, path: str, query: str = "", body: str = "") -> dict:
        timestamp = str(int(time.time()))
        message   = method + timestamp + path + query + body
        signature = hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "api-key":   self.api_key,
            "timestamp": timestamp,
            "signature": signature,
        }

    def _get(self, path: str, params: dict = None, auth: bool = False) -> dict:
        import urllib.parse
        query = ""
        if params:
            query = "?" + urllib.parse.urlencode(params)
        url = self.base_url + path + query
        headers = self._generate_signature("GET", path, query=query) if auth else {}
        resp = self.session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        import json
        url = self.base_url + path
        body_str = json.dumps(body, separators=(',', ':'))
        headers = self._generate_signature("POST", path, body=body_str)
        headers["Content-Type"] = "application/json"
        
        resp = self.session.post(url, data=body_str, headers=headers, timeout=10)
        if resp.status_code >= 400:
            try:
                err_data = resp.json()
                print(f"   ❌ Delta API Error: {err_data}")
            except:
                print(f"   ❌ HTTP {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def get_candles(self, symbol: str, resolution: str = "1m", limit: int = 1000) -> pd.DataFrame:
        """Fetch historical candles with integer-based resolutions (Delta India legacy support)."""
        # Reverting to the confirmed '1m' string format
        target_res = resolution
        res_secs_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "d": 86400}
        res_secs   = res_secs_map.get(resolution, 60)
        
        all_results = []
        # Safety: Delta India often ignores 'start/end' if 'end' is too close to 'now'
        end_ts = int(time.time()) - 300 
        # Standard Delta India batch limit is 500
        batch_size = 500 
        
        print(f"      📡 Paging {limit} candles for {symbol}...")
        while len(all_results) < limit:
            remaining = limit - len(all_results)
            current_batch = min(batch_size, remaining)
            start_ts = end_ts - (current_batch * res_secs)
            
            params = {
                "symbol": symbol,
                "resolution": target_res,
                "start": start_ts,
                "end": end_ts
            }
            
            try:
                data = self._get("/v2/history/candles", params, auth=False)
                res = data.get("result", [])
            except Exception as e:
                print(f"         ❌ API Request Failed: {e}. Retrying in 1s...")
                time.sleep(1)
                continue
            
            if not res:
                print(f"         ⚠️ End of history reached at batch {len(all_results)}.")
                break
                
            # Pivot back using the OLDEST candle in the batch (robust to API order)
            all_results = res + all_results
            oldest_res_ts = min(int(x["time"]) for x in res)
            end_ts = oldest_res_ts - 1
            
            # SHOW PROGRESS EVERY BATCH
            print(f"  Fetched {len(all_results):,} / {limit:,} candles... (Back-stepping to {pd.to_datetime(end_ts, unit='s')})")
            
            time.sleep(0.01) 

        if not all_results: return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        # Deduplicate and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        return df

    def get_orderbook(self, symbol: str) -> dict:
        """Fetch live L2 order book."""
        data = self._get(f"/v2/l2orderbook/{symbol}", auth=False)
        return data.get("result", {})

    def place_order(self, symbol: str, size: float, side: str, 
                    order_type: str = "market_order",
                    sl_pct: float = None, tp_pct: float = None) -> dict:
        """
        Place an order.
        sl_pct / tp_pct: 1.5 for 1.5%
        """
        product_id = self._resolve_product_id(symbol)
        
        # Calculate bracket prices if we have live price info
        sl_price, tp_price = None, None
        try:
            ob = self.get_orderbook(symbol)
            # Use mid price for calculation
            best_bid = float(ob['buy'][0]['price'])
            best_ask = float(ob['sell'][0]['price'])
            curr_p   = (best_bid + best_ask) / 2

            if sl_pct:
                mult = (1 - sl_pct/100) if side.lower() == "buy" else (1 + sl_pct/100)
                sl_price = round(curr_p * mult, 1)
            if tp_pct:
                mult = (1 + tp_pct/100) if side.lower() == "buy" else (1 - tp_pct/100)
                tp_price = round(curr_p * mult, 1)
        except:
            pass # Use simple order if price info fails

        payload = {
            "product_id": product_id,
            "size": int(float(size)),
            "side": side.lower(),
            "order_type": order_type
        }
        
        # 1. Place the main market order
        resp = self._post("/v2/orders", payload)
        
        # 2. Immediately attach SL/TP if requested
        if sl_price or tp_price:
            try:
                bracket_payload = {
                    "product_id": product_id,
                    "stop_loss_order": {
                        "order_type": "market_order",
                        "stop_price": str(sl_price)
                    } if sl_price else None,
                    "take_profit_order": {
                        "order_type": "market_order",
                        "stop_price": str(tp_price)
                    } if tp_price else None,
                    "bracket_stop_trigger_method": "mark_price"
                }
                # Remove None orders
                if not bracket_payload["stop_loss_order"]: del bracket_payload["stop_loss_order"]
                if not bracket_payload["take_profit_order"]: del bracket_payload["take_profit_order"]
                
                self._post("/v2/orders/bracket", bracket_payload)
                print(f"   🛡️ SL/TP Brackets Attached (SL: {sl_price}, TP: {tp_price})")
            except Exception as e:
                print(f"   ⚠️ Could not attach SL/TP: {e}")

        return resp

    def get_positions(self) -> list:
        """Get all open positions."""
        resp = self._get("/v2/positions/margined", auth=True)
        return resp.get("result", [])

    def _resolve_product_id(self, symbol: str) -> int:
        """Dynamic resolution of product_id for a symbol."""
        try:
            data = self._get("/v2/products", auth=False)
            for p in data.get("result", []):
                # Filter for Futures / Perpetuals
                if p["symbol"] == symbol and p.get("contract_type") == "perpetual_futures":
                    return int(p["id"])
        except:
            pass
        # Fallback to Testnet BTCUSD on India (84)
        return 84
