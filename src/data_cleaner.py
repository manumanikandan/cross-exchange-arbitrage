"""
Data cleaning and alignment utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def parse_timestamps(df: pd.DataFrame, timestamp_col: str, 
                    timezone: Optional[str] = None) -> pd.DataFrame:
    """
    Parse timestamp column and ensure consistent timezone.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str
        Name of timestamp column
    timezone : str, optional
        Target timezone (e.g., 'UTC')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed timestamps
    """
    df = df.copy()
    
    # Parse timestamps
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set timezone if specified
    if timezone:
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(timezone)
        else:
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(timezone)
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    return df


def split_by_exchange(df: pd.DataFrame, exchange_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into two dataframes by exchange.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exchange_col : str
        Name of exchange column
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames for ex1 and ex2
    """
    exchanges = df[exchange_col].unique()
    if len(exchanges) != 2:
        raise ValueError(f"Expected 2 exchanges, found {len(exchanges)}: {exchanges}")
    
    ex1_name, ex2_name = exchanges[0], exchanges[1]
    ex1_df = df[df[exchange_col] == ex1_name].copy()
    ex2_df = df[df[exchange_col] == ex2_name].copy()
    
    print(f"✓ Split data: {ex1_name} ({len(ex1_df):,} rows), {ex2_name} ({len(ex2_df):,} rows)")
    
    return ex1_df, ex2_df


def calculate_mid_price(df: pd.DataFrame, bid_col: str, ask_col: str) -> pd.Series:
    """
    Calculate mid price from bid and ask.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    bid_col : str
        Bid price column name
    ask_col : str
        Ask price column name
        
    Returns:
    --------
    pd.Series
        Mid prices
    """
    return (df[bid_col] + df[ask_col]) / 2


def align_data_by_second(ex1_df: pd.DataFrame, ex2_df: pd.DataFrame,
                        timestamp_col: str, symbol_col: str,
                        bid_col: str, ask_col: str,
                        depth_col: Optional[str] = None) -> pd.DataFrame:
    """
    Align data from both exchanges by second and symbol.
    
    Parameters:
    -----------
    ex1_df : pd.DataFrame
        Exchange 1 data
    ex2_df : pd.DataFrame
        Exchange 2 data
    timestamp_col : str
        Timestamp column name
    symbol_col : str
        Symbol/coin column name
    bid_col : str
        Bid price column name
    ask_col : str
        Ask price column name
    depth_col : str, optional
        Depth column name (if available)
        
    Returns:
    --------
    pd.DataFrame
        Aligned panel data with one row per (timestamp, symbol)
    """
    # Calculate mid prices
    ex1_df['mid_ex1'] = calculate_mid_price(ex1_df, bid_col, ask_col)
    ex2_df['mid_ex2'] = calculate_mid_price(ex2_df, bid_col, ask_col)
    
    # Round timestamps to nearest second
    ex1_df['t_second'] = ex1_df[timestamp_col].dt.floor('1S')
    ex2_df['t_second'] = ex2_df[timestamp_col].dt.floor('1S')
    
    # Group by second and symbol, take last value (or mean) for each second
    ex1_grouped = ex1_df.groupby(['t_second', symbol_col]).agg({
        'mid_ex1': 'last',
        bid_col: 'last',
        ask_col: 'last'
    }).reset_index()
    
    ex2_grouped = ex2_df.groupby(['t_second', symbol_col]).agg({
        'mid_ex2': 'last',
        bid_col: 'last',
        ask_col: 'last'
    }).reset_index()
    
    # Merge on timestamp and symbol
    aligned = pd.merge(
        ex1_grouped[['t_second', symbol_col, 'mid_ex1', bid_col, ask_col]],
        ex2_grouped[['t_second', symbol_col, 'mid_ex2', bid_col, ask_col]],
        on=['t_second', symbol_col],
        how='inner',
        suffixes=('_ex1', '_ex2')
    )
    
    # Rename columns for clarity
    aligned = aligned.rename(columns={
        't_second': 'timestamp',
        f'{bid_col}_ex1': 'bid_ex1',
        f'{ask_col}_ex1': 'ask_ex1',
        f'{bid_col}_ex2': 'bid_ex2',
        f'{ask_col}_ex2': 'ask_ex2'
    })
    
    # Calculate price difference
    aligned['price_diff'] = aligned['mid_ex1'] - aligned['mid_ex2']
    aligned['price_diff_pct'] = (aligned['price_diff'] / aligned['mid_ex2']) * 100
    
    # Drop rows where either exchange is missing
    aligned = aligned.dropna(subset=['mid_ex1', 'mid_ex2'])
    
    print(f"✓ Aligned data: {len(aligned):,} rows after alignment")
    print(f"  Time range: {aligned['timestamp'].min()} to {aligned['timestamp'].max()}")
    print(f"  Symbols: {aligned[symbol_col].unique().tolist()}")
    
    return aligned

