#!/usr/bin/env python3
"""
Hyperliquid Live Signal & Decision Engine
Fetches real-time 15m OHLCV, Open Interest, and Funding Rate directly from Hyperliquid.
Evaluates the audited 13-feature tree+anchor predictor, computes signal, and logs decisions.
"""
import os
import sys
import time
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))
from feature_engineering_crypto import compute_crypto_perpetual_features
from audited_predictor import AuditedCtaPredictor
from hyperliquid_ant_trader import HyperliquidAntTrader

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

def query_hl(payload: dict) -> dict:
    req = urllib.request.Request(
        HL_INFO_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def fetch_live_market_state(coin: str = "ETH", n_candles: int = 150):
    now = int(time.time() * 1000)
    # 15m * 150 = 2250 mins
    start = now - (n_candles * 15 * 60 * 1000)
    
    # 1. Fetch candles
    candles = query_hl({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "15m", "startTime": start, "endTime": now}
    })
    
    # 2. Fetch context (funding + OI)
    meta_ctx = query_hl({"type": "metaAndAssetCtxs"})
    universe = meta_ctx[0]["universe"]
    ctxs = meta_ctx[1]
    
    current_funding = 0.0001
    current_oi = 1000000.0
    for u, c in zip(universe, ctxs):
        if u["name"] == coin:
            current_funding = float(c.get("funding", 0.0001))
            current_oi = float(c.get("openInterest", 1000000.0))
            break
            
    records = []
    for c in candles:
        records.append({
            "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
            "open_interest": current_oi,
            "funding_rate": current_funding
        })
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df, current_funding, current_oi

def evaluate_decision(coin: str = "ETH", threshold: float = 0.0008):
    df, funding, oi = fetch_live_market_state(coin=coin, n_candles=150)
    if len(df) < 120:
        print(f"[WARN] Insufficient bars: {len(df)}/120")
        return None
        
    feats_df = compute_crypto_perpetual_features(df, symbol_id=1 if coin == "ETH" else 0)
    latest_bar = df.iloc[-1]
    latest_feats = feats_df.iloc[-1].to_dict()
    
    alpha = AuditedCtaPredictor.predict_one(latest_feats)
    
    decision = "HOLD / NEUTRAL"
    action = None
    if alpha > threshold:
        decision = "STRONG BUY / LONG"
        action = "BUY"
    elif alpha < -threshold:
        decision = "STRONG SELL / SHORT"
        action = "SELL"
    elif alpha > 0.0002:
        decision = "WEAK BUY (Below Threshold)"
    elif alpha < -0.0002:
        decision = "WEAK SELL (Below Threshold)"
        
    ts_str = latest_bar["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
    
    report = {
        "timestamp": ts_str,
        "coin": coin,
        "latest_close": latest_bar["close"],
        "funding_8h": funding,
        "open_interest": oi,
        "features": {
            "F00_mom64": latest_feats.get("F00"),
            "F01_mom5": latest_feats.get("F01"),
            "F02_mom20": latest_feats.get("F02"),
            "F03_mom60": latest_feats.get("F03"),
            "F04_mom120": latest_feats.get("F04"),
            "F05_eff20": latest_feats.get("F05"),
            "F06_price_oi": latest_feats.get("F06"),
            "F08_annual_carry": latest_feats.get("F08"),
            "F11_hours_to_funding": latest_feats.get("F11"),
            "F12_diurnal_sin": latest_feats.get("F12")
        },
        "alpha": alpha,
        "decision": decision,
        "action": action
    }
    return report

if __name__ == "__main__":
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"=== Hyperliquid Live Signal Evaluation ({now_str}) ===")
    
    trader = HyperliquidAntTrader()
    state = trader.get_clearinghouse_state()
    margin = state.get("marginSummary", {})
    equity = float(margin.get("accountValue", 0.0))
    print(f"[ACCOUNT] Wallet: {trader.address} | Margin Balance: ${equity:.2f} USDC")
    
    for symbol in ["ETH", "BTC"]:
        res = evaluate_decision(coin=symbol, threshold=0.0008)
        if res:
            print("\n---------------------------------------------------")
            print(f"Asset: {res['coin']} | Close: ${res['latest_close']:.2f} | Bar: {res['timestamp']}")
            f08 = res['features']['F08_annual_carry']
            print(f"Funding: {res['funding_8h'] * 100:.4f}% | Annualized Carry: {f08*100:.2f}%")
            print(f"Kaufman ER (F05): {res['features']['F05_eff20']:.4f} | 20-bar Mom (F02): {res['features']['F02_mom20']*100:.2f}%")
            print(f"Tree+Anchor Output (Alpha): {res['alpha']:.6f}")
            print(f">>> MODEL SIGNAL & DECISION: [{res['decision']}]")
            
            if res["action"] and equity > 10.0:
                print(f"[ACTION] Capital available -> Triggering ant order: {res['action']}")
            elif res["action"]:
                print(f"[SIMULATED ACTION] Signal triggered {res['action']}, but balance is ${equity:.2f} (Needs >$10 USDC for live fill)")
            else:
                print("[STATUS] Signal within neutral band, no trade needed.")
