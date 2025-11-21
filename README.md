# Cross-Exchange Arbitrage Analysis

Analysis of arbitrage opportunities between two cryptocurrency exchanges using a delta-neutral, z-score based trading strategy.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/arbitrage_analysis_cleaned.ipynb
```

## Project Structure

- `src/` - Core modules (data loading, cleaning, signal generation, backtesting)
- `notebooks/` - Analysis notebooks
- `analyze_profitable_coins.py` - Mean reversion analysis comparing profitable vs unprofitable pairs
- `results/reports/` - Generated CSV reports

## Strategy

- **Delta-neutral pair trading** with z-score entry/exit signals
- **Cost model**: ex1 (2 bps fee), ex2 (30 bps spread + 8 bps fee) ≈ 50 bps total
- **Backtesting** across 27 trading pairs with realistic execution assumptions

