"""Example script showing how to use Hyperopter for strategy optimization."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.moving_average import evaluate_strategy
from integration import create_optimizer

def evaluate_moving_average_strategy(data, params):
    """
    Evaluate a moving average crossover strategy.
    
    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters
        
    Returns:
        Strategy performance metric (Sharpe ratio)
    """
    # Convert parameters to integers
    fast_period = int(params["fast_period"])
    slow_period = int(params["slow_period"])
    
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
    """Run moving average strategy optimization."""
    # Define paths using absolute paths
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "config" / "moving_average_config.json"
    data_path = project_root / "examples" / "data" / "sample_data.csv"
    output_dir = project_root / "examples" / "results"
    
    # Print paths for debugging
    print(f"Project root: {project_root}")
    print(f"Config path: {config_path}")
    print(f"Config path exists: {config_path.exists()}")
    print(f"Data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")
    
    # Create and run optimizer
    optimizer = create_optimizer(
        config_path=str(config_path),
        data_path=str(data_path),
        strategy_evaluator=evaluate_moving_average_strategy,
        output_dir=str(output_dir)
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Get best parameters
    best_params = optimizer.get_best_parameters()
    print(f"\nBest parameters found: {best_params}")

if __name__ == "__main__":
    main()
