# NautilusTrader Crypto Perpetual Strategy Actor
from typing import Optional
import numpy as np

class MLTreeCryptoPerpetualStrategy:
    """
    NautilusTrader-compliant Strategy Actor for Crypto Perpetuals.
    Maintains causal 120-bar sliding window and executes based on tree alpha.
    """
    def __init__(self, instrument_id: str, symbol_index: int = 0, threshold: float = 0.0015):
        self.instrument_id = instrument_id
        self.symbol_index = symbol_index
        self.threshold = threshold
        self.bars = []
        self.current_position = 0.0 # -1.0, 0.0, 1.0

    def on_bar(self, bar_dict: dict) -> Optional[dict]:
        """
        Called when a 15-minute bar completes (Right-Close event).
        bar_dict contains: timestamp, open, high, low, close, volume, open_interest, funding_rate
        """
        self.bars.append(bar_dict)
        if len(self.bars) > 130:
            self.bars.pop(0)

        if len(self.bars) < 120:
            return None # Warmup phase

        # Extract features
        from feature_engineering_crypto import compute_crypto_perpetual_features
        from audited_predictor import AuditedCtaPredictor
        import pandas as pd

        df = pd.DataFrame(self.bars)
        feats_df = compute_crypto_perpetual_features(df, symbol_id=self.symbol_index)
        latest_feats = feats_df.iloc[-1].to_dict()

        # Inference
        alpha = AuditedCtaPredictor.predict_one(latest_feats)

        # Order signal generation
        target_pos = 0.0
        if alpha > self.threshold:
            target_pos = 1.0  # Long
        elif alpha < -self.threshold:
            target_pos = -1.0 # Short

        if target_pos != self.current_position:
            order = {
                "instrument_id": self.instrument_id,
                "action": "BUY" if target_pos > self.current_position else "SELL",
                "quantity": abs(target_pos - self.current_position),
                "alpha": alpha,
                "timestamp": bar_dict["timestamp"]
            }
            self.current_position = target_pos
            return order

        return None
