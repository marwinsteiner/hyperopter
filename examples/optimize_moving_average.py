"""Example script for optimizing a moving average trading strategy."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
print(f"Project root: {project_root}")
sys.path.append(project_root)

from integration import create_optimizer

def evaluate_moving_average_strategy(data: pd.DataFrame, params: dict) -> float:
    """
    Evaluate a moving average crossover strategy.
    
    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters
        
    Returns:
        Strategy performance metric (Sharpe ratio)
    """
    # Convert parameters to integers
    fast_period = int(params["fast_period"]["value"]) if isinstance(params["fast_period"], dict) else int(params["fast_period"])
    slow_period = int(params["slow_period"]["value"]) if isinstance(params["slow_period"], dict) else int(params["slow_period"])
    
    # Calculate moving averages
    fast_ma = data["Close"].rolling(window=fast_period).mean()
    slow_ma = data["Close"].rolling(window=slow_period).mean()
    
    # Generate signals
    signals = pd.Series(0, index=data.index)
    signals[fast_ma > slow_ma] = 1  # Long when fast MA crosses above slow MA
    signals[fast_ma < slow_ma] = -1  # Short when fast MA crosses below slow MA
    
    # Calculate returns
    daily_returns = data["Close"].pct_change()
    strategy_returns = signals.shift(1) * daily_returns
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return float(sharpe_ratio)

def main():
    """Run the optimization example."""
    # Get paths
    config_path = os.path.join(project_root, "config", "moving_average_config.json")
    data_path = os.path.join(project_root, "examples", "data", "sample_data.csv")
    
    print(f"Config path: {config_path}")
    print(f"Config path exists: {os.path.exists(config_path)}")
    print(f"Data path: {data_path}")
    print(f"Data path exists: {os.path.exists(data_path)}")
    print()
    
    # Create optimizer
    optimizer = create_optimizer(
        config_path=config_path,
        data_path=data_path,
        strategy_evaluator=evaluate_moving_average_strategy
    )
    
    # Run optimization
    optimizer.optimize()
    
    # Get best parameters
    best_params = optimizer.get_best_parameters()
    print("\nBest parameters found:", best_params)

if __name__ == "__main__":
    main()
