from importlib.util import find_spec

import numpy as np

from zeroth.problem import Problem

if find_spec('sklearn') is not None:
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error

else:
    ElasticNet = None
    mean_squared_error = None


def random_search(f: Problem, x0, n_iter=100):
    """
    Performs hyperparameter optimization using the Random Search algorithm.

    This algorithm explores the search space by generating random points within the search domain.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform.

    Returns:
        np.ndarray: The best solution found.
    """
    best_x = x0
    best_fx = f(best_x)

    for _ in range(n_iter):
        if f.bounded:
            x = np.random.uniform(-1, 1, f.ndim)
        else:
            x = np.random.randn(f.ndim)

        fx = f(x)

        if fx < best_fx:
            best_x = x
            best_fx = fx

    return best_x


class HyperparameterTuning(Problem):
    """
    A real-world problem of tuning hyperparameters for a machine learning model.

    The goal is to find the best `alpha` and `l1_ratio` for an ElasticNet
    regression model to minimize the mean squared error on a synthetic dataset.
    """
    def __init__(self):
        # Generate a synthetic dataset
        np.random.seed(0)
        self.X = np.random.rand(100, 2)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + np.random.randn(100) * 0.1

        # Initial guess for alpha and l1_ratio
        x0 = np.array([0.5, 0.5])
        super().__init__(x0, lb=(0,0), ub=(10,1))

    def evaluate(self, x):
        if ElasticNet is None or mean_squared_error is None:
            raise ModuleNotFoundError("sklearn is not installed")

        alpha, l1_ratio = x
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        return mean_squared_error(self.y, y_pred)


def test_random_search_on_hyperparameter_tuning():
    # skip if sklearn is not installed
    if ElasticNet is None:
        return

    problem = HyperparameterTuning()
    solution = random_search(problem, problem.x0, n_iter=100)

    # Check that the final objective function value is lower than the initial one
    loss_x0 = problem(problem.x0)
    loss_sol = problem(solution)
    assert loss_sol < loss_x0
