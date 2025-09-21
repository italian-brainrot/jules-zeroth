import numpy as np
from .coordinate_descent import coordinate_descent

def sphere(x):
    """The Sphere function, a common benchmark for optimization algorithms."""
    return np.sum(x**2)

def test_coordinate_descent_on_sphere_function():
    """
    Tests the coordinate_descent algorithm on the Sphere function.
    """
    x0 = np.array([0.5, 0.5])
    result = coordinate_descent(sphere, x0, n_iter=100, step_size=0.01)
    # The result should be close to the true minimum at [0, 0]
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)
