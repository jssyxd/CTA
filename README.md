# CTA · Institutional NautilusTrader Crypto Perpetual CTA & ML Autoresearch Pipeline

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-NautilusTrader-orange.svg)]()
[![Exchange](https://img.shields.io/badge/exchanges-Hyperliquid%20%7C%20OKX-green.svg)]()

## Overview
This repository contains the end-to-end quantitative trading infrastructure for crypto perpetual futures:
1. **Microstructure Feature Engineering**: 13-factor continuous state tracking (Momentum, Kaufman ER, Basis/Carry, Relative OI, OFI).
2. **Machine Learning Pipeline**: Transformer (Teacher) -> LightGBM/XGBoost (Student) distillation with missingness-as-a-feature anchor.
3. **Purged Walk-Forward & Monte Carlo Engine**: Strict embargo and non-overlapping validation avoiding lookahead bias.
4. **NautilusTrader Actor Execution**: Event-driven order management for Hyperliquid and OKX.
