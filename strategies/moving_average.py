"""
Moving Average Strategy Implementation

This module contains the implementation of a moving average crossover trading strategy.
Users can customize the parameters and use this as a template for their own strategies.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

def calculate_moving_averages(data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
    """
    Calculate fast and slow moving averages.
    
    Args:
        data: DataFrame with at least 'close' column
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
        
    Returns:
        DataFrame with additional MA columns
    """
    df = data.copy()
    df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
    return df

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on moving average crossovers.
    
    Args:
        data: DataFrame with 'fast_ma' and 'slow_ma' columns
        
    Returns:
        DataFrame with additional signal columns
    """
    df = data.copy()
    df['signal'] = np.where(df['fast_ma'] > df['slow_ma'], 1, -1)
    df['signal_shift'] = df['signal'].shift(1)
    df['trade'] = df['signal'] != df['signal_shift']
    return df

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate strategy returns.
    
    Args:
        data: DataFrame with 'signal' column
        
    Returns:
        DataFrame with returns columns
    """
    df = data.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def evaluate_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> float:
    """
    Evaluate the strategy performance with given parameters.
    
    Args:
        data: Input price data
        params: Strategy parameters (fast_period, slow_period)
        
    Returns:
        Strategy performance metric (Sharpe ratio)
    """
    # Calculate moving averages
    df = calculate_moving_averages(
        data, 
        fast_period=params['fast_period'],
        slow_period=params['slow_period']
    )
    
    # Generate signals and calculate returns
    df = generate_signals(df)
    df = calculate_returns(df)
    
    # Calculate Sharpe ratio
    strategy_returns = df['strategy_returns'].dropna()
    if len(strategy_returns) == 0:
        return -np.inf
        
    sharpe_ratio = np.sqrt(252) * (
        strategy_returns.mean() / strategy_returns.std()
    )
    
    return float(sharpe_ratio)
