"""
Analyze why ONDOUSDT and XCNUSDT are profitable while others are not.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path('.').resolve()))

from src.data_loader import load_market_data
from src.data_cleaner import parse_timestamps
from src.diagnostics import analyze_raw_spread
from src.backtest_engine import BacktestEngine

print("="*80)
print("ANALYZING PROFITABLE COINS")
print("="*80)

# Load data
df_raw = load_market_data('marketdata_extracted.parquet')
df_clean = parse_timestamps(df_raw, 'timestamp', timezone='UTC')

# Prepare aligned data
df_aligned = df_clean.copy()
df_aligned = df_aligned.rename(columns={
    'ex1_mid': 'mid_ex1',
    'ex2_mid': 'mid_ex2',
})
df_aligned['price_diff'] = df_aligned['mid_ex1'] - df_aligned['mid_ex2']
df_aligned['timestamp'] = df_aligned['timestamp'].dt.floor('1s')
df_aligned = df_aligned.groupby(['timestamp', 'symbol']).agg({
    'mid_ex1': 'last',
    'mid_ex2': 'last',
    'price_diff': 'last'
}).reset_index()
df_aligned = df_aligned.dropna(subset=['mid_ex1', 'mid_ex2'])

# Compare profitable vs unprofitable coins
profitable = ['ONDOUSDT', 'XCNUSDT']
unprofitable = ['TIAUSDT', 'NMRUSDT', 'EIGENUSDT']  # High opportunity but losing

print("\n1. SPREAD CHARACTERISTICS COMPARISON")
print("-" * 80)

for symbol in profitable + unprofitable:
    symbol_data = df_aligned[df_aligned['symbol'] == symbol].copy()
    
    # Calculate spread statistics
    spread = symbol_data['price_diff']
    avg_price = (symbol_data['mid_ex1'] + symbol_data['mid_ex2']) / 2
    cost_threshold = avg_price * (50.0 / 10000)
    
    # When spread exceeds cost, what happens next?
    above_cost = symbol_data[spread.abs() > cost_threshold].copy()
    
    if len(above_cost) > 0:
        # Recalculate cost threshold for above_cost subset
        above_cost_avg_price = (above_cost['mid_ex1'] + above_cost['mid_ex2']) / 2
        above_cost_threshold = above_cost_avg_price * (50.0 / 10000)
        
        # Calculate what happens in the next second
        above_cost = above_cost.sort_values('timestamp')
        above_cost['next_price_diff'] = above_cost['price_diff'].shift(-1)
        above_cost['spread_change'] = above_cost['next_price_diff'] - above_cost['price_diff']
        
        # For positive spreads (ex1 expensive), we short ex1/long ex2
        # Profit if spread decreases (mean reversion)
        positive_spreads = above_cost[above_cost['price_diff'] > above_cost_threshold]
        negative_spreads = above_cost[above_cost['price_diff'] < -above_cost_threshold]
        
        print(f"\n{symbol}:")
        print(f"  Total opportunities: {len(above_cost):,}")
        print(f"  Positive spreads (ex1 expensive): {len(positive_spreads):,}")
        print(f"  Negative spreads (ex1 cheap): {len(negative_spreads):,}")
        
        if len(positive_spreads) > 0:
            mean_reversion_pos = (positive_spreads['spread_change'] < 0).mean()
            avg_change_pos = positive_spreads['spread_change'].mean()
            print(f"  Positive spreads: {mean_reversion_pos:.1%} mean-revert, avg change: {avg_change_pos:.6f}")
        
        if len(negative_spreads) > 0:
            mean_reversion_neg = (negative_spreads['spread_change'] > 0).mean()
            avg_change_neg = negative_spreads['spread_change'].mean()
            print(f"  Negative spreads: {mean_reversion_neg:.1%} mean-revert, avg change: {avg_change_neg:.6f}")
        
        # Spread persistence
        spread_persistence = above_cost['spread_change'].abs().mean()
        print(f"  Avg spread change magnitude: {spread_persistence:.6f}")
        print(f"  Mean |spread|: {spread.abs().mean():.6f}")
        print(f"  Cost threshold: {cost_threshold.mean():.6f}")
        print(f"  Spread/cost ratio: {(spread.abs().mean() / cost_threshold.mean()):.2f}")

print("\n\n2. ORACLE TRADE ANALYSIS")
print("-" * 80)

backtest = BacktestEngine(notional=100.0, ex1_fee_bps=2.0, ex2_fee_bps=23.0)

for symbol in profitable:
    print(f"\n{symbol} (PROFITABLE):")
    oracle_trades = backtest.run_oracle_strategy(df_aligned, symbol, total_cost_bps=50.0)
    
    if len(oracle_trades) > 0:
        metrics = backtest.calculate_metrics(oracle_trades)
        
        # Analyze winning vs losing trades
        winning = oracle_trades[oracle_trades['pnl'] > 0]
        losing = oracle_trades[oracle_trades['pnl'] < 0]
        
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Avg PnL: ${metrics['avg_pnl']:.2f}")
        print(f"  Median PnL: ${metrics['median_pnl']:.2f}")
        print(f"  Avg holding time: {metrics['avg_holding_time_seconds']:.1f}s")
        
        if len(winning) > 0:
            print(f"  Winning trades:")
            print(f"    Avg PnL: ${winning['pnl'].mean():.2f}")
            print(f"    Avg holding: {winning['holding_time_seconds'].mean():.1f}s")
        
        if len(losing) > 0:
            print(f"  Losing trades:")
            print(f"    Avg PnL: ${losing['pnl'].mean():.2f}")
            print(f"    Avg holding: {losing['holding_time_seconds'].mean():.1f}s")

for symbol in unprofitable[:1]:  # Just analyze TIAUSDT
    print(f"\n{symbol} (UNPROFITABLE - High opportunities):")
    oracle_trades = backtest.run_oracle_strategy(df_aligned, symbol, total_cost_bps=50.0)
    
    if len(oracle_trades) > 0:
        metrics = backtest.calculate_metrics(oracle_trades)
        
        winning = oracle_trades[oracle_trades['pnl'] > 0]
        losing = oracle_trades[oracle_trades['pnl'] < 0]
        
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Avg PnL: ${metrics['avg_pnl']:.2f}")
        print(f"  Median PnL: ${metrics['median_pnl']:.2f}")
        print(f"  Avg holding time: {metrics['avg_holding_time_seconds']:.1f}s")
        
        if len(winning) > 0:
            print(f"  Winning trades:")
            print(f"    Avg PnL: ${winning['pnl'].mean():.2f}")
            print(f"    Avg holding: {winning['holding_time_seconds'].mean():.1f}s")
        
        if len(losing) > 0:
            print(f"  Losing trades:")
            print(f"    Avg PnL: ${losing['pnl'].mean():.2f}")
            print(f"    Avg holding: {losing['holding_time_seconds'].mean():.1f}s")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("1. Compare mean-reversion rates between profitable and unprofitable coins")
print("2. Check if profitable coins have better spread/cost ratios")
print("3. Analyze holding times - do profitable coins exit faster?")
print("4. Check if profitable coins have more persistent spreads")

