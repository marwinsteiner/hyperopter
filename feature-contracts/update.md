# Hyperopter Project Update

## Purpose

**project** -- hyperparameter optimization for trading strategies

Can supply csv data and json config file, run hyperparameter optimization, and output results.

## Current Status

Only Sharpe loss function at the moment. 

See feature contracts for details on loss functions to be implemented

Strategy CI/CD pipeline component.

## Next Steps
- more loss functions (ongoing)
- use hyperopter to optimize mean reversion strategy (1-2 months)
- greeks streaming through API available now -- need to check specifics (1-2 weeks)
    - write market data streamer for Algo-Trading-System (2 weeks-1 month)
    - implement put writing strategy directly in Algo-Trading-System (1-2 months)
    - implement 0dte strategy in Algo-Trading-System (1-2 months)
        - implement weighting engine, assign 0 weighting to 0dte strategy (1-2 months)
    - implement the mean reversion strategy once fully backtested. (3-4 months)
