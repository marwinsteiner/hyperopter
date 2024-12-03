# Hyperopter Testing Framework

This directory contains the testing framework for the Hyperopter optimization system. The tests cover all major components and their interactions.

## Components Tested

1. **Data Handler** (`test_data_handler.py`)
   - Data loading and validation
   - Preprocessing functionality
   - Train/validation splitting
   - Statistics calculation

2. **Configuration Manager** (`test_configuration_manager.py`)
   - Configuration loading and validation
   - Schema validation
   - Parameter space validation
   - Component compatibility

3. **Optimization Engine** (`test_optimization_engine.py`)
   - Multiple optimization strategies
   - Parallel trial execution
   - Results saving and loading
   - Convergence testing

## Test Data

The tests use the following types of test data:

1. **Sample Dataset**
   - Numerical features (continuous)
   - Categorical features
   - Target variable
   - Missing values for validation testing

2. **Configuration Files**
   - Valid configurations for different scenarios
   - Invalid configurations for error testing
   - Different optimization strategies

3. **Optimization Problems**
   - Rosenbrock function (challenging optimization landscape)
   - Simple quadratic function (for convergence testing)
   - Multi-dimensional parameter spaces

## Running Tests

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run all tests:
   ```bash
   python -m pytest
   ```

3. Run tests with coverage:
   ```bash
   python -m pytest --cov=./ --cov-report=html
   ```

4. Run specific test file:
   ```bash
   python -m pytest test_data_handler.py
   ```

## Test Coverage

The tests aim to cover:
- Normal operation paths
- Error handling
- Edge cases
- Component integration
- Resource management
- Performance characteristics

## Adding New Tests

When adding new tests:
1. Follow the existing test structure
2. Include both positive and negative test cases
3. Add appropriate documentation
4. Ensure compatibility with existing tests
5. Update requirements if needed
