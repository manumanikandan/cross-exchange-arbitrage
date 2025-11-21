"""
Data loading utilities for market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_market_data(file_path: str) -> pd.DataFrame:
    """
    Load market data from file. Handles various formats including compressed files.
    
    Parameters:
    -----------
    file_path : str
        Path to data file
        
    Returns:
    --------
    pd.DataFrame
        Loaded market data
    """
    file_path = Path(file_path)
    
    # Check if file is compressed
    is_compressed = file_path.suffix in ['.gz', '.gzip', '.cpgz', '.zip']
    
    # Try different loading methods
    try:
        # First, try reading as parquet (pandas handles gzip compression automatically)
        df = pd.read_parquet(file_path)
        print(f"Loaded as Parquet: {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception as e1:
        try:
            # Try with explicit gzip decompression for parquet
            import gzip
            if file_path.suffix in ['.gz', '.gzip', '.cpgz']:
                with gzip.open(file_path, 'rb') as f:
                    df = pd.read_parquet(f)
                print(f"Loaded as gzipped Parquet: {df.shape[0]:,} rows, {df.shape[1]} columns")
            else:
                raise e1
        except Exception as e2:
            try:
                # Try reading as CSV
                df = pd.read_csv(file_path)
                print(f"Loaded as CSV: {df.shape[0]:,} rows, {df.shape[1]} columns")
            except Exception as e3:
                try:
                    # Try as compressed CSV
                    import gzip
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        df = pd.read_csv(f)
                    print(f"Loaded as gzipped CSV: {df.shape[0]:,} rows, {df.shape[1]} columns")
                except Exception as e4:
                    # Try using io.BytesIO for compressed parquet
                    try:
                        import gzip
                        import io
                        with gzip.open(file_path, 'rb') as f:
                            buffer = io.BytesIO(f.read())
                        df = pd.read_parquet(buffer)
                        print(f"Loaded as compressed Parquet (via BytesIO): {df.shape[0]:,} rows, {df.shape[1]} columns")
                    except Exception as e5:
                        raise ValueError(
                            f"Could not load file '{file_path}'. Tried:\n"
                            f"  1. Direct parquet: {str(e1)[:100]}\n"
                            f"  2. Gzipped parquet: {str(e2)[:100]}\n"
                            f"  3. CSV: {str(e3)[:100]}\n"
                            f"  4. Gzipped CSV: {str(e4)[:100]}\n"
                            f"  5. BytesIO parquet: {str(e5)[:100]}"
                        )
    
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial data inspection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with inspection results
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Try to identify timestamp column
    timestamp_cols = [col for col in df.columns 
                     if any(keyword in col.lower() for keyword in ['time', 'timestamp', 'date', 't'])]
    info['potential_timestamp_cols'] = timestamp_cols
    
    # Try to identify exchange columns
    exchange_cols = [col for col in df.columns 
                    if any(keyword in col.lower() for keyword in ['ex', 'exchange'])]
    info['potential_exchange_cols'] = exchange_cols
    
    # Try to identify coin/symbol columns
    symbol_cols = [col for col in df.columns 
                  if any(keyword in col.lower() for keyword in ['coin', 'symbol', 'pair', 'asset'])]
    info['potential_symbol_cols'] = symbol_cols
    
    return info


def print_data_summary(df: pd.DataFrame, info: dict):
    """
    Print formatted data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    info : dict
        Inspection info dictionary
    """
    print("=" * 80)
    print("DATA INSPECTION SUMMARY")
    print("=" * 80)
    print(f"\nShape: {info['shape'][0]:,} rows × {info['shape'][1]} columns")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    
    print("\n" + "-" * 80)
    print("COLUMNS:")
    print("-" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        print(f"  {col:30s} | {str(dtype):15s} | {null_pct:5.2f}% missing")
    
    if info['potential_timestamp_cols']:
        print(f"\n✓ Potential timestamp columns: {info['potential_timestamp_cols']}")
    if info['potential_exchange_cols']:
        print(f"✓ Potential exchange columns: {info['potential_exchange_cols']}")
    if info['potential_symbol_cols']:
        print(f"✓ Potential symbol columns: {info['potential_symbol_cols']}")
    
    print("\n" + "-" * 80)
    print("FIRST FEW ROWS:")
    print("-" * 80)
    print(df.head())
    
    print("\n" + "-" * 80)
    print("BASIC STATISTICS:")
    print("-" * 80)
    print(df.describe())

