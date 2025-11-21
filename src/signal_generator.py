"""
Arbitrage signal generation utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_zscore_signal(df: pd.DataFrame, window: int = 300,
                           upper_threshold: float = 3.0,
                           lower_threshold: float = -3.0) -> pd.DataFrame:
    """
    Calculate z-score based arbitrage signal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aligned dataframe with price_diff column
    window : int
        Rolling window size for z-score calculation (default: 300 seconds)
    upper_threshold : float
        Upper z-score threshold for signal
    lower_threshold : float
        Lower z-score threshold for signal
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with signal columns added
    """
    df = df.copy()
    
    # Group by symbol to calculate rolling stats per symbol
    df['price_diff_mean'] = df.groupby('symbol')['price_diff'].transform(
        lambda x: x.rolling(window=window).mean()
    )
    df['price_diff_std'] = df.groupby('symbol')['price_diff'].transform(
        lambda x: x.rolling(window=window).std()
    )
    
    # Calculate z-score (handle division by zero)
    df['z_score'] = np.where(
        df['price_diff_std'] > 0,
        (df['price_diff'] - df['price_diff_mean']) / df['price_diff_std'],
        np.nan
    )
    
    # Generate signal
    # price_diff = mid_ex1 - mid_ex2
    # If z_score > threshold: ex1 is expensive relative to ex2 -> SHORT ex1, LONG ex2 (fade mispricing)
    # If z_score < -threshold: ex1 is cheap relative to ex2 -> LONG ex1, SHORT ex2 (fade mispricing)
    # Signal encoding: 1 = long ex1/short ex2 (ex1 cheap), -1 = short ex1/long ex2 (ex1 expensive)
    df['signal'] = 0
    df.loc[df['z_score'] > upper_threshold, 'signal'] = -1  # ex1 expensive -> short ex1, long ex2
    df.loc[df['z_score'] < lower_threshold, 'signal'] = 1   # ex1 cheap -> long ex1, short ex2
    
    return df


def calculate_cost_buffer(mid_price: float, total_cost_bps: float = 50.0) -> float:
    """
    Calculate minimum price difference needed to cover trading costs.
    
    Parameters:
    -----------
    mid_price : float
        Mid price (for calculating absolute cost)
    total_cost_bps : float
        Total round-trip cost in basis points
        
    Returns:
    --------
    float
        Minimum price difference in absolute terms
    """
    return mid_price * (total_cost_bps / 10000)


def filter_signal_by_cost(df: pd.DataFrame, total_cost_bps: float = 50.0,
                          safety_buffer: float = 1.2) -> pd.DataFrame:
    """
    Filter signals to only include those where price difference exceeds costs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with signals
    total_cost_bps : float
        Total round-trip cost in basis points
    safety_buffer : float
        Multiplier for safety buffer (default: 1.2 = 20% buffer)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with filtered signals
    """
    df = df.copy()
    
    # Calculate minimum required price difference
    # Use average of both mid prices for cost calculation (more accurate)
    df['avg_mid_price'] = (df['mid_ex1'] + df['mid_ex2']) / 2
    df['min_price_diff'] = df['avg_mid_price'] * (total_cost_bps / 10000) * safety_buffer
    
    # Filter signals: only trade if price difference exceeds minimum
    df['signal_filtered'] = df['signal'].copy()
    df.loc[df['price_diff'].abs() < df['min_price_diff'], 'signal_filtered'] = 0
    
    return df

