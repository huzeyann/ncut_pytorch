import pytest
import torch
import numpy as np

@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed

@pytest.fixture
def small_feature_matrix():
    """Create a small feature matrix for testing."""
    return torch.rand(100, 10)

@pytest.fixture
def medium_feature_matrix():
    """Create a medium-sized feature matrix for testing."""
    return torch.rand(1000, 20)

@pytest.fixture
def large_feature_matrix():
    """Create a large feature matrix for testing."""
    return torch.rand(5000, 30)

@pytest.fixture
def device():
    """Return the device to use for testing."""
    return 'cpu'  # Use CPU for testing to ensure consistency

@pytest.fixture
def ncut_params():
    """Return a dictionary of parameters for Ncut."""
    return {
        'n_eig': 10,
        'track_grad': False,
        'd_gamma': 0.1,
        'device': 'cpu',
    }