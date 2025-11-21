"""
Backtesting engine for arbitrage strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    symbol: str
    direction: int  # 1 = long ex1/short ex2, -1 = short ex1/long ex2
    entry_price_ex1: float
    entry_price_ex2: float
    notional: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price_ex1: Optional[float] = None
    exit_price_ex2: Optional[float] = None
    pnl: Optional[float] = None
    
    def calculate_pnl(self, ex1_fee_bps: float = 2.0, ex2_fee_bps: float = 23.0):
        """
        Calculate PnL for the trade.
        
        ex2_fee_bps includes: 15 bps half-spread + 8 bps fee = 23 bps per side
        
        For delta-neutral trading: trade equal USD notional on both exchanges.
        """
        if self.exit_price_ex1 is None or self.exit_price_ex2 is None:
            return None
        
        # Calculate units for delta-neutral position (equal USD notional on both sides)
        units_ex1 = self.notional / self.entry_price_ex1
        units_ex2 = self.notional / self.entry_price_ex2
        
        # Entry costs (fee as percentage of notional traded)
        entry_cost_ex1 = self.notional * (ex1_fee_bps / 10000)
        entry_cost_ex2 = self.notional * (ex2_fee_bps / 10000)
        
        # Exit costs (fee as percentage of notional at exit)
        exit_notional_ex1 = units_ex1 * self.exit_price_ex1
        exit_notional_ex2 = units_ex2 * self.exit_price_ex2
        exit_cost_ex1 = exit_notional_ex1 * (ex1_fee_bps / 10000)
        exit_cost_ex2 = exit_notional_ex2 * (ex2_fee_bps / 10000)
        
        # PnL calculation (delta-neutral: equal USD notional on both sides)
        if self.direction == 1:  # Long ex1, short ex2
            pnl_ex1 = (self.exit_price_ex1 - self.entry_price_ex1) * units_ex1
            pnl_ex2 = (self.entry_price_ex2 - self.exit_price_ex2) * units_ex2  # Short position
            pnl = pnl_ex1 + pnl_ex2
        else:  # Short ex1, long ex2
            pnl_ex1 = (self.entry_price_ex1 - self.exit_price_ex1) * units_ex1  # Short position
            pnl_ex2 = (self.exit_price_ex2 - self.entry_price_ex2) * units_ex2
            pnl = pnl_ex1 + pnl_ex2
        
        # Subtract all costs (entry + exit for both legs)
        pnl -= (entry_cost_ex1 + entry_cost_ex2 + exit_cost_ex1 + exit_cost_ex2)
        
        self.pnl = pnl
        return pnl


class BacktestEngine:
    """Simple backtesting engine for arbitrage strategy."""
    
    def __init__(self, notional: float = 100.0, 
                 ex1_fee_bps: float = 2.0,
                 ex2_fee_bps: float = 23.0):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        notional : float
            USD notional per trade
        ex1_fee_bps : float
            Exchange 1 fee in basis points
        ex2_fee_bps : float
            Exchange 2 fee in basis points (half-spread + fee)
        """
        self.notional = notional
        self.ex1_fee_bps = ex1_fee_bps
        self.ex2_fee_bps = ex2_fee_bps
        self.trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}  # symbol -> Trade
        
    def run_backtest(self, df: pd.DataFrame, symbol: str, 
                    signal_col: str = 'signal_filtered',
                    timestamp_col: str = 'timestamp',
                    symbol_col: str = 'symbol',
                    exit_threshold: float = 0.0) -> pd.DataFrame:
        """
        Run backtest for a single symbol.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with signals and prices
        symbol : str
            Symbol to backtest
        signal_col : str
            Column name for signals
        timestamp_col : str
            Timestamp column name
        symbol_col : str
            Symbol column name
        exit_threshold : float
            Z-score threshold for exit (default: 0 = exit when signal crosses zero)
            
        Returns:
        --------
        pd.DataFrame
            Results dataframe
        """
        symbol_data = df[df[symbol_col] == symbol].copy().sort_values(timestamp_col).reset_index(drop=True)
        
        # Process with proper execution timing: signal at time t, execute at time t+1
        # This avoids look-ahead bias by ensuring we don't use prices from the same bar as signal
        for i in range(len(symbol_data)):
            current_row = symbol_data.iloc[i]
            current_signal = current_row[signal_col]
            current_time = current_row[timestamp_col]
            
            # Check if we have an open trade for this symbol
            if symbol in self.open_trades:
                trade = self.open_trades[symbol]
                
                # Check exit condition: signal crosses zero or opposite direction
                if (current_signal == 0) or (current_signal * trade.direction < 0):
                    # Close the trade at current bar prices (execution at time t after seeing signal)
                    trade.exit_time = current_time
                    trade.exit_price_ex1 = current_row['mid_ex1']
                    trade.exit_price_ex2 = current_row['mid_ex2']
                    trade.calculate_pnl(self.ex1_fee_bps, self.ex2_fee_bps)
                    self.trades.append(trade)
                    del self.open_trades[symbol]
            
            # Check entry condition: use signal from current bar, execute at next bar (if available)
            # For the last bar, execute immediately (realistic: signal triggers immediate execution)
            if symbol not in self.open_trades and current_signal != 0:
                # Execute at next bar to avoid look-ahead bias
                if i < len(symbol_data) - 1:
                    # Use next bar's prices for execution
                    next_row = symbol_data.iloc[i + 1]
                    entry_time = next_row[timestamp_col]
                    entry_price_ex1 = next_row['mid_ex1']
                    entry_price_ex2 = next_row['mid_ex2']
                else:
                    # Last bar: execute at current prices (signal triggers immediate execution)
                    entry_time = current_time
                    entry_price_ex1 = current_row['mid_ex1']
                    entry_price_ex2 = current_row['mid_ex2']
                
                # Open new trade
                trade = Trade(
                    entry_time=entry_time,
                    symbol=symbol,
                    direction=current_signal,
                    entry_price_ex1=entry_price_ex1,
                    entry_price_ex2=entry_price_ex2,
                    notional=self.notional
                )
                self.open_trades[symbol] = trade
        
        # Close any remaining open trades at the end
        for symbol, trade in list(self.open_trades.items()):
            last_row = symbol_data.iloc[-1]
            trade.exit_time = last_row[timestamp_col]
            trade.exit_price_ex1 = last_row['mid_ex1']
            trade.exit_price_ex2 = last_row['mid_ex2']
            trade.calculate_pnl(self.ex1_fee_bps, self.ex2_fee_bps)
            self.trades.append(trade)
        
        self.open_trades.clear()
        
        # Convert trades to DataFrame
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_price_ex1': t.entry_price_ex1,
                    'entry_price_ex2': t.entry_price_ex2,
                    'exit_price_ex1': t.exit_price_ex1,
                    'exit_price_ex2': t.exit_price_ex2,
                    'notional': t.notional,
                    'pnl': t.pnl,
                    'holding_time_seconds': (t.exit_time - t.entry_time).total_seconds() if t.exit_time else None
                }
                for t in self.trades if t.symbol == symbol
            ])
            return trades_df
        else:
            return pd.DataFrame()
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        trades_df : pd.DataFrame
            DataFrame with trade results
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'max_drawdown': 0.0,
                'avg_holding_time_seconds': 0.0
            }
        
        trades_df = trades_df.copy()
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
        
        metrics = {
            'total_trades': len(trades_df),
            'total_pnl': trades_df['pnl'].sum(),
            'win_rate': (trades_df['pnl'] > 0).mean(),
            'avg_pnl': trades_df['pnl'].mean(),
            'median_pnl': trades_df['pnl'].median(),
            'std_pnl': trades_df['pnl'].std(),
            'max_pnl': trades_df['pnl'].max(),
            'min_pnl': trades_df['pnl'].min(),
            'max_drawdown': trades_df['drawdown'].min(),
            'avg_holding_time_seconds': trades_df['holding_time_seconds'].mean(),
            'median_holding_time_seconds': trades_df['holding_time_seconds'].median(),
        }
        
        return metrics
    
    def run_oracle_strategy(self, df: pd.DataFrame, symbol: str,
                           total_cost_bps: float = 50.0,
                           timestamp_col: str = 'timestamp',
                           symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        Run "oracle" arbitrage strategy: trade whenever |spread| > cost, close next bar.
        
        This is a diagnostic test to check if the data contains any exploitable edge.
        If this oracle strategy is negative, the data has no edge after costs.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with prices
        symbol : str
            Symbol to backtest
        total_cost_bps : float
            Total round-trip cost in basis points
        timestamp_col : str
            Timestamp column name
        symbol_col : str
            Symbol column name
            
        Returns:
        --------
        pd.DataFrame
            Results dataframe
        """
        symbol_data = df[df[symbol_col] == symbol].copy().sort_values(timestamp_col).reset_index(drop=True)
        
        # Calculate spread and cost threshold
        symbol_data['spread'] = symbol_data['mid_ex1'] - symbol_data['mid_ex2']
        symbol_data['avg_price'] = (symbol_data['mid_ex1'] + symbol_data['mid_ex2']) / 2
        symbol_data['cost_threshold'] = symbol_data['avg_price'] * (total_cost_bps / 10000)
        
        # Oracle signal: trade whenever |spread| > cost
        symbol_data['oracle_signal'] = 0
        symbol_data.loc[symbol_data['spread'] > symbol_data['cost_threshold'], 'oracle_signal'] = -1  # Short ex1, long ex2
        symbol_data.loc[symbol_data['spread'] < -symbol_data['cost_threshold'], 'oracle_signal'] = 1   # Long ex1, short ex2
        
        # Run backtest with oracle signals
        return self.run_backtest(symbol_data, symbol, signal_col='oracle_signal',
                                timestamp_col=timestamp_col, symbol_col=symbol_col)

