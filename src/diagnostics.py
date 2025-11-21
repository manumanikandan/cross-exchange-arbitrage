"""
Market diagnostics and visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


def plot_price_comparison(df: pd.DataFrame, symbol: str, symbol_col: str,
                         timestamp_col: str, save_path: Optional[str] = None):
    """
    Plot mid prices of both exchanges over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe
    symbol : str
        Symbol to plot
    symbol_col : str
        Symbol column name
    timestamp_col : str
        Timestamp column name
    save_path : str, optional
        Path to save figure
    """
    symbol_data = df[df[symbol_col] == symbol].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(symbol_data[timestamp_col], symbol_data['mid_ex1'], 
           label='Exchange 1', alpha=0.7, linewidth=1)
    ax.plot(symbol_data[timestamp_col], symbol_data['mid_ex2'], 
           label='Exchange 2', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mid Price')
    ax.set_title(f'{symbol}: Mid Price Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_price_difference(df: pd.DataFrame, symbol: str, symbol_col: str,
                         timestamp_col: str, save_path: Optional[str] = None):
    """
    Plot price difference over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe
    symbol : str
        Symbol to plot
    symbol_col : str
        Symbol column name
    timestamp_col : str
        Timestamp column name
    save_path : str, optional
        Path to save figure
    """
    symbol_data = df[df[symbol_col] == symbol].copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(symbol_data[timestamp_col], symbol_data['price_diff'], 
           alpha=0.7, linewidth=1, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price Difference (ex1 - ex2)')
    ax.set_title(f'{symbol}: Price Difference Over Time')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def calculate_basic_stats(df: pd.DataFrame, symbol: str, symbol_col: str) -> dict:
    """
    Calculate basic statistics for price difference and returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe
    symbol : str
        Symbol to analyze
    symbol_col : str
        Symbol column name
        
    Returns:
    --------
    dict
        Dictionary with statistics
    """
    symbol_data = df[df[symbol_col] == symbol].copy()
    
    # Calculate returns
    symbol_data['ret_ex1'] = np.log(symbol_data['mid_ex1'] / symbol_data['mid_ex1'].shift(1))
    symbol_data['ret_ex2'] = np.log(symbol_data['mid_ex2'] / symbol_data['mid_ex2'].shift(1))
    
    stats = {
        'price_diff_mean': symbol_data['price_diff'].mean(),
        'price_diff_std': symbol_data['price_diff'].std(),
        'price_diff_min': symbol_data['price_diff'].min(),
        'price_diff_max': symbol_data['price_diff'].max(),
        'returns_correlation': symbol_data['ret_ex1'].corr(symbol_data['ret_ex2']),
        'ret_ex1_mean': symbol_data['ret_ex1'].mean(),
        'ret_ex1_std': symbol_data['ret_ex1'].std(),
        'ret_ex2_mean': symbol_data['ret_ex2'].mean(),
        'ret_ex2_std': symbol_data['ret_ex2'].std(),
    }
    
    return stats


def plot_histograms(df: pd.DataFrame, symbol: str, symbol_col: str,
                   save_path: Optional[str] = None):
    """
    Plot histograms of price difference and returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe
    symbol : str
        Symbol to plot
    symbol_col : str
        Symbol column name
    save_path : str, optional
        Path to save figure
    """
    symbol_data = df[df[symbol_col] == symbol].copy()
    
    # Calculate returns
    symbol_data['ret_ex1'] = np.log(symbol_data['mid_ex1'] / symbol_data['mid_ex1'].shift(1))
    symbol_data['ret_ex2'] = np.log(symbol_data['mid_ex2'] / symbol_data['mid_ex2'].shift(1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Price difference histogram
    axes[0, 0].hist(symbol_data['price_diff'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Price Difference (ex1 - ex2)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{symbol}: Price Difference Distribution')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Returns histogram - ex1
    axes[0, 1].hist(symbol_data['ret_ex1'].dropna(), bins=50, alpha=0.7, 
                   color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Log Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'{symbol}: Exchange 1 Returns Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Returns histogram - ex2
    axes[1, 0].hist(symbol_data['ret_ex2'].dropna(), bins=50, alpha=0.7, 
                   color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Log Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{symbol}: Exchange 2 Returns Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot of returns
    axes[1, 1].scatter(symbol_data['ret_ex1'], symbol_data['ret_ex2'], 
                      alpha=0.3, s=1)
    axes[1, 1].set_xlabel('Exchange 1 Returns')
    axes[1, 1].set_ylabel('Exchange 2 Returns')
    axes[1, 1].set_title(f'{symbol}: Returns Scatter Plot')
    corr = symbol_data['ret_ex1'].corr(symbol_data['ret_ex2'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=axes[1, 1].transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_raw_spread(df: pd.DataFrame, symbol: str, symbol_col: str,
                       total_cost_bps: float = 50.0) -> dict:
    """
    Analyze raw spread distribution to check if arbitrage edge exists.
    
    This is a critical diagnostic: if |spread| rarely exceeds costs,
    no strategy will be profitable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe
    symbol : str
        Symbol to analyze
    symbol_col : str
        Symbol column name
    total_cost_bps : float
        Total round-trip cost in basis points
        
    Returns:
    --------
    dict
        Dictionary with spread statistics
    """
    symbol_data = df[df[symbol_col] == symbol].copy()
    
    # Calculate spread (ex1 - ex2)
    spread = symbol_data['price_diff']
    avg_price = (symbol_data['mid_ex1'] + symbol_data['mid_ex2']) / 2
    
    # Calculate cost threshold in absolute terms
    cost_threshold = avg_price * (total_cost_bps / 10000)
    
    # Calculate statistics
    stats = {
        'mean_spread': spread.mean(),
        'mean_abs_spread': spread.abs().mean(),
        'std_spread': spread.std(),
        'min_spread': spread.min(),
        'max_spread': spread.max(),
        'cost_threshold_mean': cost_threshold.mean(),
        'fraction_above_cost': (spread.abs() > cost_threshold).mean(),
        'fraction_above_cost_1_2x': (spread.abs() > cost_threshold * 1.2).mean(),
        'num_opportunities': (spread.abs() > cost_threshold).sum(),
        'total_seconds': len(symbol_data)
    }
    
    return stats
