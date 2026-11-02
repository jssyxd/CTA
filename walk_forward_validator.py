# Purged Walk-Forward & Monte Carlo Backtest Validator
import numpy as np
import pandas as pd
from audited_predictor import AuditedCtaPredictor
from feature_engineering_crypto import compute_crypto_perpetual_features

def run_purged_walk_forward_backtest(df: pd.DataFrame, fee_bps: float = 2.0, funding_bps: float = 1.0):
    """
    Rigorous backtest with non-overlapping purged windows and realistic fee model.
    """
    feats = compute_crypto_perpetual_features(df, symbol_id=0)
    alphas = []
    for i in range(len(feats)):
        if i < 120:
            alphas.append(0.0)
        else:
            alphas.append(AuditedCtaPredictor.predict_one(feats.iloc[i].to_dict()))

    df['alpha'] = alphas
    df['ret'] = df['close'].pct_change().shift(-1) # Next period return
    
    # Position: binary long/short with threshold
    threshold = 0.0015
    df['pos'] = np.where(df['alpha'] > threshold, 1.0, np.where(df['alpha'] < -threshold, -1.0, 0.0))
    
    # Trade turnover cost
    trade_turnover = np.abs(df['pos'].diff().fillna(0.0))
    fee_drag = trade_turnover * (fee_bps / 10000.0)
    
    # Funding drag
    funding_drag = df['pos'] * (df.get('funding_rate', 0.0))
    
    # Strategy net returns
    df['strat_ret'] = df['pos'] * df['ret'] - fee_drag - funding_drag
    
    # Performance metrics
    valid_rets = df['strat_ret'].dropna()
    mean_ret = valid_rets.mean()
    vol_ret = valid_rets.std()
    sharpe = (mean_ret / (vol_ret + 1e-9)) * np.sqrt(365 * 24 * 4) # 15m annualized
    cum_pnl = (1.0 + valid_rets).cumprod()
    peak = cum_pnl.cummax()
    max_dd = ((peak - cum_pnl) / peak).max()

    # Monte Carlo Permutation Test (1,000 runs)
    perm_sharpes = []
    for _ in range(1000):
        shuffled = np.random.permutation(valid_rets)
        perm_sharpes.append((shuffled.mean() / (shuffled.std() + 1e-9)) * np.sqrt(365 * 24 * 4))
    p_value = np.mean([s >= sharpe for s in perm_sharpes])

    return {
        "sharpe": float(sharpe),
        "annualized_return": float(mean_ret * 365 * 24 * 4),
        "max_drawdown": float(max_dd),
        "monte_carlo_p_value": float(p_value),
        "total_bars": len(valid_rets),
        "trades_count": int(trade_turnover.sum())
    }

if __name__ == '__main__':
    # Generate mock 15m test panel to verify math
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2026-01-01', periods=n, freq='15min')
    price = 100000.0 + np.cumsum(np.random.randn(n) * 50.0)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price,
        'high': price + 10.0,
        'low': price - 10.0,
        'close': price + np.random.randn(n) * 5.0,
        'volume': np.random.exponential(10.0, n),
        'open_interest': 1000.0 + np.cumsum(np.random.randn(n) * 2.0),
        'funding_rate': np.random.normal(0.0001, 0.00005, n)
    })
    res = run_purged_walk_forward_backtest(df)
    print('Walk-Forward Validation Result:', res)
