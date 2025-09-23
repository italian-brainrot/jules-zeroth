from importlib.util import find_spec

import numpy as np

from zeroth.problem import Problem

if find_spec('sklearn') is not None:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_breast_cancer
else:
    GaussianProcessRegressor = None
    Matern = None
    SVC = None
    cross_val_score = None
    load_breast_cancer = None


def bayesian_optimization(f: Problem, n_iter=20, kappa=2.576):
    """
    Performs hyperparameter optimization using Bayesian Optimization.

    This algorithm uses a Gaussian Process to model the objective function and an
    acquisition function to decide where to sample next.

    Args:
        f (function): The objective function to minimize.
        n_iter (int): The number of iterations to perform.
        kappa (float): The exploration-exploitation trade-off parameter for the UCB acquisition function.

    Returns:
        np.ndarray: The best solution found.
    """
    if GaussianProcessRegressor is None:
        raise ModuleNotFoundError("scikit-learn is not installed")

    X_sample = np.array([f.x0])
    Y_sample = np.array([f(f.x0)])

    # Initialize Gaussian Process regressor
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

    for _ in range(n_iter):
        # Update Gaussian Process regressor
        gpr.fit(X_sample, Y_sample)

        # Propose next sample point by maximizing the acquisition function (Upper Confidence Bound)
        if f.bounded:
            x_domain = np.random.uniform(-1, 1, size=(1000, f.ndim))
        else:
            x_domain = np.random.randn(1000, f.ndim)

        mean, std = gpr.predict(x_domain, return_std=True)
        ucb = mean - kappa * std  # We are minimizing, so we subtract

        next_sample = x_domain[np.argmin(ucb)]

        # Evaluate the objective function at the next sample point
        y_next = f(next_sample)

        # Add the new sample to our history
        X_sample = np.vstack([X_sample, next_sample])
        Y_sample = np.append(Y_sample, y_next)

    # Return the best solution found
    best_idx = np.argmin(Y_sample)
    return X_sample[best_idx]


class SvmHyperparameterTuning(Problem):
    """
    A real-world problem of tuning hyperparameters for a Support Vector Machine (SVM) model.

    The goal is to find the best `C` and `gamma` for an SVM with an RBF kernel
    to maximize the cross-validation score on the breast cancer dataset.
    """
    def __init__(self):
        if load_breast_cancer is None:
            raise ModuleNotFoundError("scikit-learn is not installed")

        self.X, self.y = load_breast_cancer(return_X_y=True)

        # Initial guess for C and gamma
        x0 = np.array([1.0, 0.1])
        super().__init__(x0, lb=(1e-6, 1e-6), ub=(100, 1))

    def evaluate(self, x):
        if SVC is None or cross_val_score is None:
            raise ModuleNotFoundError("scikit-learn is not installed")

        C, gamma = x
        model = SVC(C=C, gamma=gamma, random_state=0)

        # We want to maximize the score, so we minimize its negative
        return -np.mean(cross_val_score(model, self.X, self.y, cv=3))


def test_bayesian_optimization_on_svm_hyperparameter_tuning():
    # skip if sklearn is not installed
    if SVC is None:
        return

    problem = SvmHyperparameterTuning()
    solution = bayesian_optimization(problem, n_iter=10)

    # Check that the final objective function value is lower than the initial one
    loss_x0 = problem(problem.x0)
    loss_sol = problem(solution)
    assert loss_sol < loss_x0
