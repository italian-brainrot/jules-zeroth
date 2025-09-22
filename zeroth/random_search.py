import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from zeroth.problem import Problem


def random_search(f: Problem, x0, n_iter=1000, step_size=0.1):
    """
    Performs unconstrained optimization using the Random Search algorithm.

    This algorithm explores the search space by generating random steps
    from the current best point. If a new point yields a better objective
    function value, it becomes the new best point.

    A real-world example of its use is in hyperparameter tuning for
    machine learning models, where the objective function is the validation
    error and the parameters are the model's hyperparameters.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform.
        step_size (float): The standard deviation of the random steps.

    Returns:
        np.ndarray: The parameters that minimize the objective function.
    """
    best_x = x0
    best_fx = f(best_x)

    for _ in range(n_iter):
        candidate_x = best_x + np.random.normal(scale=step_size, size=x0.shape)
        candidate_fx = f(candidate_x)

        if candidate_fx < best_fx:
            best_x = candidate_x
            best_fx = candidate_fx

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
        super().__init__(x0)

    def f(self, x):
        alpha, l1_ratio = x
        # Ensure alpha is non-negative and l1_ratio is in [0, 1]
        alpha = abs(alpha)
        l1_ratio = np.clip(l1_ratio, 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        return mean_squared_error(self.y, y_pred)


def test_random_search_on_hyperparameter_tuning():
    problem = HyperparameterTuning()
    best_params = random_search(problem, problem.x0, n_iter=100, step_size=0.1)

    # Check that the final objective function value is lower than the initial one
    initial_mse = problem(problem.x0)
    final_mse = problem(best_params)
    assert final_mse < initial_mse
