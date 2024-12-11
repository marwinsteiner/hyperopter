"""Tests for the profit/loss function."""

import pytest
import pandas as pd
import numpy as np

from loss_functions import ProfitLossFunction


@pytest.fixture
def sample_trades():
    """Create sample trade data for testing."""
    return pd.DataFrame({
        "pnl": [100, -50, 75, -25, 200],  # Total: 300
        "position_size": [1000, 2000, 1500, 1000, 2500]  # Total: 8000
    })


def test_initialization():
    """Test profit/loss function initialization."""
    # Test valid initialization
    pnl = ProfitLossFunction(initial_capital=10000.0, transaction_fee=0.001)
    assert pnl.initial_capital == 10000.0
    assert pnl.transaction_fee == 0.001
    assert pnl.min_trades == 10
    assert pnl.direction == "maximize"
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        ProfitLossFunction(initial_capital=0)
    with pytest.raises(ValueError):
        ProfitLossFunction(initial_capital=-1000)
    with pytest.raises(ValueError):
        ProfitLossFunction(transaction_fee=-0.001)
    with pytest.raises(ValueError):
        ProfitLossFunction(min_trades=0)


def test_calculation(sample_trades):
    """Test P&L calculation with fees."""
    pnl = ProfitLossFunction(
        initial_capital=10000.0,
        transaction_fee=0.001,  # 0.1% fee
        min_trades=5
    )
    
    # Calculate with position sizes
    result = pnl(
        trade_data=sample_trades,
        position_sizes=sample_trades["position_size"]
    )
    
    # Expected calculations:
    # Total P&L: 300
    # Total volume: 8000
    # Total fees: 8000 * 0.001 = 8
    # Net P&L: 300 - 8 = 292
    assert result == pytest.approx(292.0)
    
    # Check metadata
    metadata = pnl.get_metadata()
    assert metadata["total_pnl"] == 300.0
    assert metadata["total_fees"] == pytest.approx(8.0)
    assert metadata["net_pnl"] == pytest.approx(292.0)
    assert metadata["n_trades"] == 5
    assert metadata["avg_pnl_per_trade"] == pytest.approx(58.4)
    assert metadata["total_volume"] == 8000.0
    assert metadata["roi"] == pytest.approx(0.0292)


def test_insufficient_trades():
    """Test handling of insufficient trades."""
    pnl = ProfitLossFunction(min_trades=10)
    
    # Create data with too few trades
    data = pd.DataFrame({
        "pnl": [100, -50, 75],
        "position_size": [1000, 2000, 1500]
    })
    
    with pytest.raises(ValueError, match="Insufficient trades"):
        pnl(data)


def test_missing_data():
    """Test handling of missing data."""
    pnl = ProfitLossFunction()
    
    # Create data without pnl column
    data = pd.DataFrame({
        "position_size": [1000, 2000, 1500]
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        pnl(data)


def test_edge_cases(sample_trades):
    """Test edge cases and boundary conditions."""
    pnl = ProfitLossFunction(
        initial_capital=10000.0,
        transaction_fee=0.0,  # No fees
        min_trades=1
    )
    
    # Test with no fees
    result = pnl(sample_trades)
    assert result == 300.0  # Raw P&L with no fees
    
    # Test with all losing trades
    losing_trades = pd.DataFrame({
        "pnl": [-100, -50, -75, -25, -200]
    })
    result = pnl(losing_trades)
    assert result == -450.0
    
    # Test with zero P&L trades
    zero_trades = pd.DataFrame({
        "pnl": [0, 0, 0, 0, 0]
    })
    result = pnl(zero_trades)
    assert result == 0.0


def test_memory_usage():
    """Test memory usage with large datasets."""
    # Create large dataset (100k trades)
    n_trades = 100_000
    large_data = pd.DataFrame({
        "pnl": np.random.normal(0, 100, n_trades),
        "position_size": np.random.uniform(1000, 10000, n_trades)
    })
    
    pnl = ProfitLossFunction(min_trades=100)
    
    # Measure memory before
    mem_before = large_data.memory_usage().sum()
    
    # Run calculation
    _ = pnl(large_data)
    
    # Measure memory after
    mem_after = large_data.memory_usage().sum()
    
    # Memory usage should not increase significantly
    assert mem_after <= mem_before * 1.1  # Allow 10% overhead
