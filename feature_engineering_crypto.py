# Feature Engineering Pipeline for Crypto Perpetuals
import numpy as np
import pandas as pd

def compute_crypto_perpetual_features(df: pd.DataFrame, symbol_id: int = 0) -> pd.DataFrame:
    """
    Converts 15m OHLCV + Open Interest + Funding Rate into the 13 audited CTA features.
    Strictly causal (no lookahead).
    """
    feats = pd.DataFrame(index=df.index)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    oi = df.get('open_interest', pd.Series(np.ones(len(df)))).values
    funding = df.get('funding_rate', pd.Series(np.zeros(len(df)))).values

    # F00: momentum_64
    feats['F00'] = close - np.roll(close, 64)
    feats.loc[:64, 'F00'] = np.nan

    # F01: daily_mom5 (5 bars)
    feats['F01'] = (close - np.roll(close, 5)) / (np.roll(close, 5) + 1e-9)
    feats.loc[:5, 'F01'] = np.nan

    # F02: daily_mom20 (20 bars)
    feats['F02'] = (close - np.roll(close, 20)) / (np.roll(close, 20) + 1e-9)
    feats.loc[:20, 'F02'] = np.nan

    # F03: daily_mom60 (60 bars)
    feats['F03'] = (close - np.roll(close, 60)) / (np.roll(close, 60) + 1e-9)
    feats.loc[:60, 'F03'] = np.nan

    # F04: daily_mom120 (120 bars)
    feats['F04'] = (close - np.roll(close, 120)) / (np.roll(close, 120) + 1e-9)
    feats.loc[:120, 'F04'] = np.nan

    # F05: daily_eff20 (Kaufman Efficiency Ratio over 20 bars)
    direction = np.abs(close - np.roll(close, 20))
    volatility = np.zeros_like(close)
    abs_diff = np.abs(close - np.roll(close, 1))
    for i in range(20, len(close)):
        volatility[i] = np.sum(abs_diff[i-19:i+1])
    feats['F05'] = np.where(volatility > 1e-9, direction / np.maximum(volatility, 1e-9), 0.0)
    feats.loc[:20, 'F05'] = np.nan

    # F06: daily_price_oi (price diff * OI diff normalized)
    d_price = close - np.roll(close, 1)
    d_oi = oi - np.roll(oi, 1)
    feats['F06'] = np.tanh(d_price * d_oi / (close * np.maximum(oi, 1.0) * 1e-4 + 1e-9))

    # F07: product_id (categorical)
    feats['F07'] = symbol_id

    # F08: term_carry5 (annualized funding rate / basis)
    feats['F08'] = funding * (365.0 * 24.0 / 8.0)

    # F09: term_carry_change5
    feats['F09'] = feats['F08'] - np.roll(feats['F08'], 5)
    feats.loc[:5, 'F09'] = np.nan

    # F10: relative_to_main_oi (relative to 20-bar median OI)
    oi_roll_median = pd.Series(oi).rolling(20, min_periods=1).median().values
    feats['F10'] = np.clip(oi / (oi_roll_median + 1e-9), 0.0, 3.0)

    # F11: days_to_delivery_month -> mapped to hours_to_next_funding
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], utc=True)
        hours = dt.dt.hour.values + dt.dt.minute.values / 60.0
        feats['F11'] = 8.0 - (hours % 8.0)
        # F12: delivery_sin -> diurnal solar cycle
        feats['F12'] = np.sin(2.0 * np.pi * hours / 24.0)
    else:
        feats['F11'] = 4.0
        feats['F12'] = 0.0

    return feats
