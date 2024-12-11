"""Tests for the Sortino ratio loss function."""

import pytest
import pandas as pd
import numpy as np

from loss_functions import SortinoRatioLoss


@pytest.fixture
def sample_returns():
    """Create sample return data for testing."""
    return pd.DataFrame({
        "pnl": [100, -50, 75, -25, 200, -100, 150, -75, 50, -25]
    })


def test_initialization():
    """Test Sortino ratio loss function initialization."""
    # Test valid initialization
    loss = SortinoRatioLoss(mar=0.0, frequency="daily")
    assert loss.mar == 0.0
    assert loss.frequency == "daily"
    assert loss.direction == "minimize"
    assert loss.annualization_factor == 252  # Raw factor, not sqrt
    
    # Test hourly frequency
    loss = SortinoRatioLoss(frequency="hourly")
    assert loss.annualization_factor == 252 * 24
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        SortinoRatioLoss(frequency="weekly")


def test_calculation(sample_returns):
    """Test Sortino ratio calculation."""
    loss = SortinoRatioLoss(mar=0.0, frequency="daily")
    ratio = loss(sample_returns)
    
    # Verify metadata
    metadata = loss.get_metadata()
    assert "mean_return" in metadata
    assert "downside_deviation" in metadata
    assert metadata["n_trades"] == 10
    assert metadata["n_downside_trades"] == 5  # Number of negative returns
    
    # Test with positive MAR
    loss = SortinoRatioLoss(mar=50.0)
    ratio_high_mar = loss(sample_returns)
    assert ratio_high_mar > ratio  # Higher MAR should result in lower Sortino ratio (but higher loss)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with all positive returns
    positive_returns = pd.DataFrame({
        "pnl": [100, 50, 75, 25, 200]
    })
    loss = SortinoRatioLoss(mar=0.0)  # Explicitly set MAR to 0
    ratio = loss(positive_returns)
    assert ratio == -np.inf  # All returns above MAR (perfect strategy = minimum loss)
    
    # Test with all negative returns
    negative_returns = pd.DataFrame({
        "pnl": [-100, -50, -75, -25, -200]
    })
    ratio = loss(negative_returns)
    assert ratio > 0  # Should be positive (bad strategy = high loss)
    
    # Test with constant returns at MAR
    constant_returns = pd.DataFrame({
        "pnl": [0, 0, 0, 0, 0]  # Equal to MAR
    })
    ratio = loss(constant_returns)
    assert ratio == np.inf  # All returns equal to MAR (worst strategy = maximum loss)
    
    # Test with single return
    single_return = pd.DataFrame({
        "pnl": [100]
    })
    ratio = loss(single_return)
    assert ratio == -np.inf  # Single return above MAR (perfect strategy = minimum loss)


def test_error_handling():
    """Test error handling."""
    loss = SortinoRatioLoss()
    
    # Test missing pnl column
    invalid_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=5)
    })
    with pytest.raises(ValueError, match="Missing required column"):
        loss(invalid_data)
    
    # Test empty DataFrame
    empty_data = pd.DataFrame({"pnl": []})
    with pytest.raises(ValueError, match="trade_data is empty"):
        loss(empty_data)


def test_frequency_impact():
    """Test impact of different frequencies."""
    returns = pd.DataFrame({
        "pnl": [100, -50, 75, -25, 200]
    })
    
    daily_loss = SortinoRatioLoss(frequency="daily")
    hourly_loss = SortinoRatioLoss(frequency="hourly")
    
    daily_ratio = daily_loss(returns)
    hourly_ratio = hourly_loss(returns)
    
    # Hourly ratio should be sqrt(24) times larger due to annualization
    assert np.isclose(
        hourly_ratio / daily_ratio,
        np.sqrt(24),
        rtol=1e-10
    )
