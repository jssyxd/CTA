# Hyperliquid EIP-712 Order Signing & Ant-Trading Engine
import os
import time
import json
import urllib.request
from eth_account import Account

class HyperliquidAntTrader:
    def __init__(self, env_path="W:/database/CTA/.env"):
        self.load_env(env_path)
        self.private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "")
        self.account = Account.from_key(self.private_key)
        self.address = self.account.address
        self.base_url = "https://api.hyperliquid.xyz"

    def load_env(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.startswith("#") and "=" in line:
                        k, v = line.strip().split("=", 1)
                        os.environ[k] = v

    def get_clearinghouse_state(self):
        req = urllib.request.Request(
            self.base_url + "/info",
            data=json.dumps({"type": "clearinghouseState", "user": self.address}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def get_mid_price(self, coin="ETH"):
        req = urllib.request.Request(
            self.base_url + "/info",
            data=json.dumps({"type": "allMids"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            mids = json.loads(resp.read().decode())
            return float(mids.get(coin, 0.0))

    def prepare_ant_order(self, coin="ETH", is_buy=True, size_eth=0.005):
        mid = self.get_mid_price(coin)
        limit_px = round(mid * 0.985 if is_buy else mid * 1.015, 1)
        nonce = int(time.time() * 1000)
        return {
            "asset": 1 if coin == "ETH" else 0,
            "isBuy": is_buy,
            "limitPx": limit_px,
            "sz": size_eth,
            "reduceOnly": False,
            "orderType": {"limit": {"tif": "Alo"}},
            "nonce": nonce
        }

if __name__ == "__main__":
    trader = HyperliquidAntTrader()
    print("[ANT-TRADER] Account Public Address:", trader.address)
    state = trader.get_clearinghouse_state()
    margin = state.get("marginSummary", {})
    equity = float(margin.get("accountValue", 0.0))
    print("[ACCOUNT STATUS] Current Account Value:", equity, "USDC")
    mid_eth = trader.get_mid_price("ETH")
    print("[MARKET] Live Hyperliquid ETH Mid Price:", mid_eth)
    order = trader.prepare_ant_order(coin="ETH", is_buy=True, size_eth=0.005)
    print("[READY] Ant-Sized Order Pre-flight Payload:", json.dumps(order, indent=2))
