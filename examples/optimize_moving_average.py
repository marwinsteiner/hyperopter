"""Example script showing how to use Hyperopter for strategy optimization."""

from pathlib import Path
from strategies.moving_average import evaluate_strategy
from integration import create_optimizer

def main():
    """Run moving average strategy optimization."""
    # Define paths
    current_dir = Path(__file__).parent
    config_path = current_dir.parent / "config" / "moving_average_config.json"
    data_path = current_dir / "data" / "sample_data.csv"
    output_dir = current_dir / "results"
    
    # Create and run optimizer
    optimizer = create_optimizer(
        config_path=str(config_path),
        data_path=str(data_path),
        strategy_evaluator=evaluate_strategy,
        output_dir=str(output_dir)
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Get best parameters
    best_params = optimizer.get_best_parameters()
    print(f"\nBest parameters found: {best_params}")

if __name__ == "__main__":
    main()
