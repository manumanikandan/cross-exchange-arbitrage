"""
Utility functions for arbitrage analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
        
    Returns:
    --------
    pd.Series
        Log returns
    """
    return np.log(prices / prices.shift(1))


def calculate_zscore(series: pd.Series, window: int = 300) -> pd.Series:
    """
    Calculate rolling z-score.
    
    Parameters:
    -----------
    series : pd.Series
        Input series
    window : int
        Rolling window size (default: 300 seconds = 5 minutes)
        
    Returns:
    --------
    pd.Series
        Z-scores
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std


def basis_points_to_decimal(bps: float) -> float:
    """
    Convert basis points to decimal.
    
    Parameters:
    -----------
    bps : float
        Basis points
        
    Returns:
    --------
    float
        Decimal value
    """
    return bps / 10000


def calculate_cross_correlation(series1: pd.Series, series2: pd.Series, 
                                max_lag: int = 10) -> pd.DataFrame:
    """
    Calculate cross-correlation between two series at different lags.
    
    Parameters:
    -----------
    series1 : pd.Series
        First series
    series2 : pd.Series
        Second series
    max_lag : int
        Maximum lag to test
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag and correlation values
    """
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = series1.shift(lag).corr(series2)
        elif lag > 0:
            corr = series1.corr(series2.shift(lag))
        else:
            corr = series1.corr(series2)
        results.append({'lag': lag, 'correlation': corr})
    
    return pd.DataFrame(results)


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series,
                                  window: int = 300) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Parameters:
    -----------
    series1 : pd.Series
        First series
    series2 : pd.Series
        Second series
    window : int
        Rolling window size in seconds
        
    Returns:
    --------
    pd.Series
        Rolling correlation values
    """
    # Create a DataFrame to use rolling().corr()
    df = pd.DataFrame({'series1': series1, 'series2': series2})
    return df['series1'].rolling(window=window).corr(df['series2'])

