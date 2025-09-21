import numpy as np
from .random_search import random_search

def sphere(x):
    """The Sphere function, a common benchmark for optimization algorithms."""
    return np.sum(x**2)

def test_random_search_on_sphere_function():
    """
    Tests the random_search algorithm on the Sphere function.
    """
    x0 = np.array([0.5, 0.5])
    # Set a seed for reproducibility
    np.random.seed(42)
    result = random_search(sphere, x0, n_iter=2000, step_size=0.1)
    # The result should be close to the true minimum at [0, 0]
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)
