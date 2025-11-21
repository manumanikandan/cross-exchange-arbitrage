# Cross-Exchange Arbitrage Analysis

Analysis of arbitrage opportunities between two cryptocurrency exchanges using a delta-neutral, z-score based trading strategy. Backtesting across 27 trading pairs reveals concentrated profitability in 4 coins (+$633.50) while 23 coins lost (-$1,202.06), demonstrating that exchange relationship dynamics and balanced mean-reversion are critical determinants of arbitrage success.

📄 **Full Report**: See [`ARBITRAGE_ANALYSIS_REPORT.pdf`](ARBITRAGE_ANALYSIS_REPORT.pdf) for detailed methodology, findings, and insights.

## Key Findings

- **4 profitable coins** generated **+$633.50** total PnL (led by ONDOUSDT at +$392.33 with 69.16% win rate)
- **23 losing coins** generated **-$1,202.06** total PnL
- **Net portfolio PnL**: -$568.56 across 11,458 trades
- **Critical insight**: Profitability depends on balanced mean-reversion (40-55% in both directions) and low spread persistence

## Strategy Overview

- **Delta-neutral pair trading** with z-score entry/exit signals (±3.0 threshold)
- **Cost model**: ex1 (2 bps fee), ex2 (30 bps spread + 8 bps fee) ≈ 50 bps total round-trip
- **Position sizing**: Equal USD notional ($100 long + $100 short) eliminates directional risk
- **Average holding time**: 1-2 seconds per trade

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/arbitrage_analysis_cleaned.ipynb
```

## Project Structure

- `src/` - Core modules (data loading, cleaning, signal generation, backtesting)
- `notebooks/arbitrage_analysis_cleaned.ipynb` - Main analysis notebook
- `analyze_profitable_coins.py` - Mean reversion analysis comparing profitable vs unprofitable pairs
- `results/reports/` - Generated CSV reports (trades, performance metrics, sample data)
- `ARBITRAGE_ANALYSIS_REPORT.pdf` - Comprehensive analysis report

## Why Some Coins Work and Others Don't

**Profitable coins** (ONDOUSDT, KAITOUSDT, XCNUSDT, PNUTUSDT) exhibit:
- Balanced mean-reversion: 40-55% convergence in both positive and negative spread directions
- Low spread persistence: Spreads close quickly (1-2 seconds)
- High win rates: 42-69% vs 30% average for losing coins

**Unprofitable coins** fail due to:
- Asymmetric mean-reversion: High convergence in one direction but low in the other
- High spread persistence: Spreads stay wide or continue diverging
- Low win rates: Insufficient to overcome 50 bps transaction costs

## Exchange Relationship Analysis

Both exchanges react **simultaneously** to market information (correlation ~0.98-0.99 at lag 0), indicating efficient price discovery. Arbitrage opportunities exist but are quickly exploited, making mean-reversion characteristics the key differentiator between profitable and unprofitable pairs.

