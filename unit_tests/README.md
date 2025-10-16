# Unit Tests for ncut_pytorch

This directory contains unit tests for the `ncut_pytorch` package. The tests are written using the pytest framework and cover the core functionality, edge cases, and error handling of the package.

## Test Structure

The tests are organized into the following files:

- `test_ncut.py`: Tests for the `Ncut` class and its methods
- `test_nystrom_ncut.py`: Tests for the `ncut_fn` function and related functions
- `test_kway_ncut.py`: Tests for the `kway_ncut` and `axis_align` functions
- `test_biased_ncut.py`: Tests for the `bias_ncut_soft` and `get_mask_and_heatmap` functions
- `test_visualize_utils.py`: Tests for the visualization utilities
- `test_edge_cases.py`: Tests for edge cases and error handling

## Setup

To run the tests, you need to install the development dependencies:

```bash
pip install ncut-pytorch[dev]
```

This will install pytest and other dependencies needed for testing.

## Running the Tests

To run all the tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest unit_tests/test_ncut.py
```

To run a specific test:

```bash
pytest unit_tests/test_ncut.py::TestNcut::test_fit
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=ncut_pytorch
```

For a more detailed HTML report:

```bash
pytest --cov=ncut_pytorch --cov-report=html
```

This will create a `htmlcov` directory with the coverage report.

## Writing New Tests

When writing new tests, follow these guidelines:

1. Use the pytest framework and the existing test structure
2. Use fixtures from `conftest.py` for common test data
3. Test both normal operation and edge cases
4. Use descriptive test names and docstrings
5. Keep tests independent of each other

## Continuous Integration

These tests can be integrated into a CI/CD pipeline to ensure that changes to the codebase don't break existing functionality. For example, you can use GitHub Actions to run the tests on every push and pull request.

## Test Data

The tests use synthetic data generated in the fixtures defined in `conftest.py`. This ensures that the tests are reproducible and don't depend on external data.

## Troubleshooting

If you encounter issues with the tests:

1. Make sure you have installed all the dependencies
2. Check that you're running the tests from the project root directory
3. Try running with verbose output: `pytest -v`
4. If a specific test is failing, run it in isolation to debug