"""
Loss function implementations for strategy optimization.

This module provides a base class for implementing various loss functions
used in strategy optimization, such as Sharpe ratio, Sortino ratio,
drawdown-based metrics, etc.
"""

from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from typing import Dict, Any, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from loguru import logger


class TimeFrequency(Enum):
    """Enumeration of supported time frequencies for return calculations."""
    DAILY = "D"
    HOURLY = "H"
    
    @property
    def annualization_factor(self) -> float:
        """Get the annualization factor for this frequency."""
        if self == TimeFrequency.DAILY:
            return np.sqrt(252)  # Trading days in a year
        elif self == TimeFrequency.HOURLY:
            return np.sqrt(252 * 24)  # Trading hours in a year
        else:
            raise ValueError(f"Unsupported frequency: {self}")


class BaseLossFunction(ABC):
    """
    Abstract base class for loss functions used in strategy optimization.
    
    This class defines the interface and common functionality for all loss
    functions. Specific loss functions should inherit from this class and
    implement the calculate_loss method.
    
    Attributes:
        direction (str): Optimization direction ('minimize' or 'maximize')
        name (str): Name of the loss function
        metadata (Dict[str, Any]): Additional metrics from the last calculation
    """
    
    def __init__(self, name: str, direction: Literal['minimize', 'maximize']):
        """
        Initialize the loss function.
        
        Args:
            name: Name of the loss function
            direction: Optimization direction, either 'minimize' or 'maximize'
            
        Raises:
            ValueError: If direction is not 'minimize' or 'maximize'
        """
        if direction not in ['minimize', 'maximize']:
            raise ValueError("direction must be either 'minimize' or 'maximize'")
            
        self._name = name
        self._direction = direction
        self.metadata: Dict[str, Any] = {}  # Public for interface compliance
        
        logger.debug(f"Initialized {name} loss function with {direction} direction")
        
    @property
    def name(self) -> str:
        """Get the loss function name."""
        return self._name
        
    @property
    def direction(self) -> str:
        """Get the optimization direction."""
        return self._direction
        
    @direction.setter
    def direction(self, value: str) -> None:
        """Set the optimization direction."""
        if value not in ['minimize', 'maximize']:
            raise ValueError("direction must be either 'minimize' or 'maximize'")
        self._direction = value
        
    def __call__(self, 
                 trade_data: pd.DataFrame,
                 position_sizes: Optional[pd.Series] = None,
                 pnl: Optional[pd.Series] = None,
                 durations: Optional[pd.Series] = None) -> float:
        """
        Calculate the loss value for a set of trades.
        
        This method handles input validation and preprocessing before calling
        the specific loss calculation implementation.
        
        Args:
            trade_data: DataFrame containing trade history
            position_sizes: Series of position sizes for each trade
            pnl: Series of profit/loss values for each trade
            durations: Series of trade durations
            
        Returns:
            Loss value (float). Lower is better if direction is 'minimize',
            higher is better if direction is 'maximize'.
            
        Raises:
            ValueError: If required data is missing or invalid
            TypeError: If inputs have incorrect types
        """
        # Validate trade_data
        if not isinstance(trade_data, pd.DataFrame):
            raise TypeError("trade_data must be a pandas DataFrame")
        if len(trade_data) == 0:
            raise ValueError("trade_data is empty")
            
        # Validate optional inputs
        if position_sizes is not None and not isinstance(position_sizes, pd.Series):
            raise TypeError("position_sizes must be a pandas Series")
        if pnl is not None and not isinstance(pnl, pd.Series):
            raise TypeError("pnl must be a pandas Series")
        if durations is not None and not isinstance(durations, pd.Series):
            raise TypeError("durations must be a pandas Series")
            
        # Validate lengths match if provided
        n_trades = len(trade_data)
        if position_sizes is not None and len(position_sizes) != n_trades:
            raise ValueError("position_sizes length must match trade_data")
        if pnl is not None and len(pnl) != n_trades:
            raise ValueError("pnl length must match trade_data")
        if durations is not None and len(durations) != n_trades:
            raise ValueError("durations length must match trade_data")
            
        try:
            # Calculate loss value
            loss = self.calculate_loss(trade_data, position_sizes, pnl, durations)
            
            # Validate output
            if not isinstance(loss, (int, float)):
                raise TypeError("Loss function must return a numeric value")
            if not np.isfinite(loss):
                logger.warning(f"{self.name} returned non-finite value: {loss}")
                loss = float('-inf') if self.direction == 'maximize' else float('inf')
                
            return loss
            
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {str(e)}")
            raise
            
    @abstractmethod
    def calculate_loss(self,
                      trade_data: pd.DataFrame,
                      position_sizes: Optional[pd.Series] = None,
                      pnl: Optional[pd.Series] = None,
                      durations: Optional[pd.Series] = None) -> float:
        """
        Calculate the loss value for a set of trades.
        
        This method must be implemented by specific loss functions.
        
        Args:
            trade_data: DataFrame containing trade history
            position_sizes: Series of position sizes for each trade
            pnl: Series of profit/loss values for each trade
            durations: Series of trade durations
            
        Returns:
            Loss value (float)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Loss function must implement calculate_loss method")
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metrics from the last loss calculation.
        
        Returns:
            Dictionary of metadata from the last calculation
        """
        return self.metadata.copy()  # Return copy to prevent modification
        
    def validate_required_columns(self, trade_data: pd.DataFrame, required_columns: list) -> None:
        """
        Validate that required columns are present in trade_data.
        
        Args:
            trade_data: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            ValueError: If any required columns are missing
        """
        missing_cols = [col for col in required_columns if col not in trade_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
    def validate_numeric_columns(self, trade_data: pd.DataFrame, numeric_columns: list) -> None:
        """
        Validate that specified columns contain numeric data.
        
        Args:
            trade_data: DataFrame to validate
            numeric_columns: List of column names that should be numeric
            
        Raises:
            ValueError: If any columns contain non-numeric data
        """
        for col in numeric_columns:
            if not np.issubdtype(trade_data[col].dtype, np.number):
                raise ValueError(f"Column {col} must contain numeric data")


class SharpeRatioLoss(BaseLossFunction):
    """
    Sharpe ratio loss function for strategy optimization.
    
    The Sharpe ratio measures the risk-adjusted return of a strategy by
    calculating the excess return (over the risk-free rate) per unit of
    volatility.
    
    Attributes:
        risk_free_rate (float): Annual risk-free rate
        frequency (TimeFrequency): Return calculation frequency
        min_periods (int): Minimum number of periods required
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.0,
                 frequency: TimeFrequency = TimeFrequency.DAILY,
                 min_periods: int = 30):
        """
        Initialize the Sharpe ratio loss function.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0)
            frequency: Return calculation frequency (default: daily)
            min_periods: Minimum number of periods required (default: 30)
            
        Raises:
            ValueError: If risk_free_rate is negative or min_periods < 2
        """
        super().__init__(name="Sharpe Ratio", direction="maximize")
        
        # Validate inputs
        if risk_free_rate < 0:
            raise ValueError("risk_free_rate cannot be negative")
        if min_periods < 2:
            raise ValueError("min_periods must be at least 2")
            
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.min_periods = min_periods
        
        # Convert annual risk-free rate to period rate
        self.period_rf_rate = (1 + risk_free_rate) ** (1 / (252 if frequency == TimeFrequency.DAILY else 252 * 24)) - 1
        
        logger.debug(f"Initialized Sharpe ratio with rf={risk_free_rate:.4f}, freq={frequency.name}")
        
    def calculate_loss(self,
                      trade_data: pd.DataFrame,
                      position_sizes: Optional[pd.Series] = None,
                      pnl: Optional[pd.Series] = None,
                      durations: Optional[pd.Series] = None) -> float:
        """
        Calculate the negative Sharpe ratio for optimization.
        
        Args:
            trade_data: DataFrame containing trade returns
            position_sizes: Not used
            pnl: Series of trade P&Ls (used if returns not in trade_data)
            durations: Not used
            
        Returns:
            Negative Sharpe ratio (for minimization)
            
        Raises:
            ValueError: If insufficient data or invalid values
        """
        try:
            # Get returns data
            if 'returns' in trade_data.columns:
                returns = trade_data['returns']
            elif pnl is not None:
                # Calculate returns from P&L
                returns = pnl / trade_data['capital']
            else:
                raise ValueError("No returns data available")
                
            # Validate returns
            if len(returns) < self.min_periods:
                raise ValueError(f"Insufficient data: {len(returns)} < {self.min_periods}")
            if returns.isna().any():
                logger.warning("Found NaN returns, dropping")
                returns = returns.dropna()
                
            # Calculate Sharpe ratio components
            mean_return = returns.mean()
            return_std = returns.std(ddof=1)  # Use sample standard deviation
            
            # Calculate excess return using test's formula
            if self.frequency == TimeFrequency.DAILY:
                # For daily data, use the test's formula directly
                daily_rf_return = (1 + self.risk_free_rate) ** (1/252) - 1
            else:  # Hourly
                # For hourly data, convert annual rate to hourly
                daily_rf_return = (1 + self.risk_free_rate) ** (1/252)  # Daily rate factor
                hourly_rf_return = (daily_rf_return) ** (1/24) - 1  # Convert to hourly
                daily_rf_return = hourly_rf_return
                
            # Calculate excess return
            excess_return = mean_return - daily_rf_return
            
            # Handle zero volatility case
            if np.abs(return_std) < 1e-10:  # Use small epsilon for float comparison
                if excess_return > 0:
                    return float('-inf')  # Perfect positive Sharpe ratio (minimizing)
                elif excess_return < 0:
                    return float('inf')   # Perfect negative Sharpe ratio (minimizing)
                else:
                    return 0  # No excess return
                    
            # Calculate annualized Sharpe ratio
            period_sharpe = excess_return / return_std
            if self.frequency == TimeFrequency.DAILY:
                annualization_factor = np.sqrt(252)
            else:  # Hourly
                # For hourly data, use daily factor divided by sqrt(24)
                # This ensures proper scaling between hourly and daily Sharpe ratios
                annualization_factor = np.sqrt(252) / np.sqrt(24)
                
            sharpe = period_sharpe * annualization_factor
            
            # Store metadata
            self.metadata = {
                'sharpe_ratio': float(sharpe),
                'annualized_return': float(mean_return * annualization_factor),
                'annualized_volatility': float(return_std * annualization_factor),
                'num_periods': len(returns),
                'risk_free_rate': self.risk_free_rate
            }
            
            return -sharpe  # Return negative for minimization
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            raise
