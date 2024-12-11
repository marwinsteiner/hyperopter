# Feature: Calmar Ratio Loss Function  

## Input/Output Contract  
Input:  
- Returns and equity curve from data_handler.py  
- Configuration from configuration_manager.py  
Output:  
- Negative Calmar ratio (float)  
- Optimization direction: 'minimize'  
- Complex risk metrics  

## Dependencies  
- loss_functions_base.py  
- drawdown_loss.py  
- data_handler.py  
- configuration_manager.py  
- memory_manager.py  

## Integration Points  
- optimization_engine.py  
- parallel_optimizer.py  
- results_manager.py  
- logging_system.py  

## Success Criteria  
- Accurate ratio calculation  
- Proper period handling  
- < 250ms execution time  
- Memory efficient  

## Validation Steps  
1. Unit tests in tests/test_calmar_ratio_loss.py  
2. Component calculation verification  
3. Memory usage profiling  
4. Performance benchmarking  
5. Integration testing  