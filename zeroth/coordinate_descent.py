import numpy as np

from zeroth.problem import Problem


def coordinate_descent(f: Problem, n_iter: int = 100, step_size: float = 0.1):
    """
    Performs unconstrained optimization using the Coordinate Descent algorithm.

    This algorithm minimizes a function by iteratively performing approximate
    minimization along coordinate directions. At each iteration, it minimizes
    the function with respect to a single coordinate, keeping the others fixed.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform over all coordinates.
        step_size (float): The size of the step to take in each coordinate direction.

    Returns:
        np.ndarray: The best solution found.
    """
    x = f.x0.copy()

    for _ in range(n_iter):
        for i in range(n_dims):
            # Create vectors for positive and negative steps
            step_positive = np.zeros(f.ndim)
            step_positive[i] = step_size
            step_negative = np.zeros(f.ndim)
            step_negative[i] = -step_size

            # Evaluate function at current point and after taking steps
            fx = f(x)
            fx_positive = f(x + step_positive)
            fx_negative = f(x + step_negative)

            # Choose the best of the three points
            if fx_positive < fx:
                x += step_positive
            elif fx_negative < fx:
                x += step_negative

    return x


class LinearSystem(Problem):
    """
    A real-world problem of solving a system of linear equations Ax = b.

    This is framed as a minimization problem: minimize ||Ax - b||^2.
    """
    def __init__(self):
        # Generate a synthetic linear system
        np.random.seed(0)
        self.A = np.random.rand(10, 5)
        self.b = np.random.rand(10)

        # Initial guess for x
        x0 = np.zeros(5)
        super().__init__(x0)

    def evaluate(self, x):
        return np.linalg.norm(self.A @ x - self.b)**2


def test_coordinate_descent_on_linear_system():
    problem = LinearSystem()
    solution = coordinate_descent(problem, problem.x0, n_iter=100, step_size=0.01)

    # Check that the final objective function value is lower than the initial one
    initial_residual = problem(problem.x0)
    final_residual = problem(solution)
    assert final_residual < initial_residual
