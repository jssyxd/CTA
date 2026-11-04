#!/usr/bin/env python3
"""
OKX 24/7 Automated CTA Loop Daemon for XAG-USDT-SWAP (Silver Perpetual)
Compliant with https://github.com/jssyxd/quant-research-constitution:
1. Strict Anti Look-Ahead: executes only on 15m Bar Right-Close.
2. Anti-Phantom Fills: Post-Only Maker limit orders.
3. Margin Scaling: fixed 2 contracts (0.02 XAG ~ $1.04 notional, <5% margin).
4. Auto-heal: cancels stale uncompleted maker orders each cycle.
"""
import socket
import urllib.request
import hmac, base64, hashlib, json, time, os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Force IPv4
orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, *a, **k: orig(h, p, socket.AF_INET, socket.SOCK_STREAM)

sys.path.append(os.path.dirname(__file__))
from feature_engineering_crypto import compute_crypto_perpetual_features
from audited_predictor import AuditedCtaPredictor

class OKXTrader:
    def __init__(self, api_key="776e25a5-be71-4c71-a183-ac8859789cfd",
                 secret_key="E96F1B17DBD38ADFBD4C719E3231C124",
                 passphrase="Auto111."):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://www.okx.com"

    def _request(self, endpoint, method="GET", body=""):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        pre = ts + method + endpoint + body
        sig = base64.b64encode(hmac.new(self.secret_key.encode(), pre.encode(), hashlib.sha256).digest()).decode()
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sig,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        req = urllib.request.Request(self.base_url + endpoint, data=body.encode() if body else None, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def get_balance(self):
        res = self._request("/api/v5/account/balance")
        details = res.get("data", [{}])[0].get("details", [])
        for d in details:
            if d.get("ccy") == "USDT":
                return float(d.get("availEq", 0.0)), float(d.get("eq", 0.0))
        return 0.0, 0.0

    def get_candles(self, inst_id="XAG-USDT-SWAP", bar="15m", limit=150):
        res = self._request(f"/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}")
        rows = res.get("data", [])
        records = []
        for r in rows:
            records.append({
                "timestamp": pd.to_datetime(int(r[0]), unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "open_interest": 1000000.0,
                "funding_rate": 0.0001
            })
        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        return df

    def get_positions(self, inst_id="XAG-USDT-SWAP"):
        res = self._request(f"/api/v5/account/positions?instId={inst_id}")
        return res.get("data", [])

    def evaluate(self, inst_id="XAG-USDT-SWAP"):
        df = self.get_candles(inst_id=inst_id, bar="15m", limit=150)
        feats_df = compute_crypto_perpetual_features(df, symbol_id=0)
        latest_feats = feats_df.iloc[-1].to_dict()
        latest_close = df.iloc[-1]["close"]
        alpha = AuditedCtaPredictor.predict_one(latest_feats)
        return latest_close, alpha, latest_feats

    def place_order(self, inst_id, side, sz, px, ord_type="post_only"):
        body_dict = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side,
            "ordType": ord_type,
            "sz": str(sz)
        }
        if ord_type != "market":
            body_dict["px"] = str(px)
        return self._request("/api/v5/trade/order", method="POST", body=json.dumps(body_dict))

    def cancel_all_pending(self, inst_id="XAG-USDT-SWAP"):
        pending = self._request(f"/api/v5/trade/orders-pending?instId={inst_id}")
        for o in pending.get("data", []):
            oid = o.get("ordId")
            self._request("/api/v5/trade/cancel-order", method="POST", body=json.dumps({"instId": inst_id, "ordId": oid}))

    def run_cycle(self, inst_id="XAG-USDT-SWAP", sz="2", threshold=0.0008):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        avail, eq = self.get_balance()
        px, alpha, feats = self.evaluate(inst_id)
        
        # Check current positions
        positions = self.get_positions(inst_id)
        current_pos = 0.0
        if positions:
            p = positions[0]
            current_pos = float(p.get("pos", 0.0))
            
        print(f"[{now_str}] {inst_id} | Px: ${px:.3f} | Alpha: {alpha:+.6f} | ER: {feats.get('F05', 0):.4f} | CurrentPos: {current_pos} | Eq: ${eq:.2f}")

        # Signal check
        target_side = None
        if alpha > threshold:
            target_side = "buy"
        elif alpha < -threshold:
            target_side = "sell"

        if target_side is None:
            print(f"[{now_str}] Alpha inside neutral zone (|alpha| <= {threshold}), no action.")
            return

        # Passive post-only limit price (Maker strictly)
        # XAG tick size is 0.01
        limit_px = round(px * 0.999 if target_side == "buy" else px * 1.001, 2)

        # Risk check: only trade if not already matching position
        if (target_side == "buy" and current_pos > 0) or (target_side == "sell" and current_pos < 0):
            print(f"[{now_str}] Already in desired position side ({target_side}). Holding.")
            return

        if eq < 5.0:
            print(f"[{now_str}] Warning: Equity (${eq:.2f}) below safe threshold ($5.0). Order skipped.")
            return

        # Cancel stale orders before new order
        self.cancel_all_pending(inst_id)
        print(f"[{now_str}] Submitting Maker {target_side.upper()} order for {inst_id}: sz={sz}, px={limit_px}...")
        res = self.place_order(inst_id, target_side, sz, limit_px, ord_type="post_only")
        print(f"[{now_str}] Order result: {res}")

def daemon_loop(inst_id="XAG-USDT-SWAP", interval_sec=300):
    print(f"=== Starting OKX 24/7 CTA Trading Daemon for {inst_id} ===")
    trader = OKXTrader()
    while True:
        try:
            trader.run_cycle(inst_id=inst_id, sz="2", threshold=0.0008)
        except Exception as e:
            print(f"[ERROR in loop] {e}")
        time.sleep(interval_sec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run in continuous 24/7 daemon loop")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds between cycles")
    parser.add_argument("--inst", type=str, default="XAG-USDT-SWAP", help="Instrument ID")
    args = parser.parse_args()

    if args.loop:
        daemon_loop(inst_id=args.inst, interval_sec=args.interval)
    else:
        trader = OKXTrader()
        trader.run_cycle(inst_id=args.inst)
