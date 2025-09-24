import numpy as np

from zeroth.problem import Problem


def spsa_2(f: Problem, n_iter=1000, a=0.01, c=0.01, A=100, alpha=0.602, gamma=0.101):
    """
    Minimizes a function using the second-order SPSA (2-SPSA) algorithm.

    Args:
        f (Problem): The problem to solve.
        n_iter (int): The number of iterations.
        a (float): SPSA parameter.
        c (float): SPSA parameter.
        A (float): SPSA stability constant.
        alpha (float): SPSA parameter.
        gamma (float): SPSA parameter.

    Returns:
        np.ndarray: The best solution found.
    """
    x = f.x0
    hessian_avg = np.zeros((f.ndim, f.ndim))
    for k in range(n_iter):
        ak = a / (k + 1.0 + A) ** alpha
        ck = c / (k + 1.0) ** gamma

        # Generate two random perturbation vectors
        delta1 = np.random.choice([-1, 1], size=f.ndim)
        delta2 = np.random.choice([-1, 1], size=f.ndim)

        # Estimate the gradient
        y_plus = f(x + ck * delta1)
        y_minus = f(x - ck * delta1)
        grad = (y_plus - y_minus) / (2 * ck) * delta1

        # Estimate the Hessian using function evaluations
        y_pp = f(x + ck * delta1 + ck * delta2)
        y_pm = f(x + ck * delta1 - ck * delta2)
        y_mp = f(x - ck * delta1 + ck * delta2)
        y_mm = f(x - ck * delta1 - ck * delta2)

        hess_est_num = (y_pp - y_pm - y_mp + y_mm) / (4 * ck**2)
        hess_est = hess_est_num * np.outer(delta1, delta2)

        # Update the Hessian average using a weighted average
        w_k = 1.0 / (k + 1.0)
        hessian_avg = (1 - w_k) * hessian_avg + w_k * hess_est

        # Symmetrize the Hessian estimate
        hessian = (hessian_avg + hessian_avg.T) / 2

        # Use the pseudo-inverse for stability with ill-conditioned matrices
        hessian_inv = np.linalg.pinv(hessian + 1e-8 * np.eye(f.ndim))

        # Update the solution
        x = x - ak * hessian_inv @ grad

        if f.bounded:
            x = np.clip(x, -1, 1)

    return x


class MulticollinearityRegression(Problem):
    """
    A regression problem with multicollinearity, where some predictor
    variables are highly correlated. This makes the Hessian of the loss
    function ill-conditioned. Second-order methods like 2-SPSA are
    well-suited for such problems as they can account for the curvature
    of the loss landscape.
    """
    def __init__(self, n_features=10, n_samples=100, noise=0.1):
        np.random.seed(0)
        # Generate synthetic data
        X = np.random.rand(n_samples, n_features)
        # Introduce multicollinearity
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)
        X[:, 3] = X[:, 2] + 0.1 * np.random.randn(n_samples)

        # True coefficients
        self.true_coeffs = np.random.randn(n_features)
        y = X @ self.true_coeffs + noise * np.random.randn(n_samples)

        self.X = X
        self.y = y

        x0 = np.zeros(n_features)
        super().__init__(x0)

    def evaluate(self, x):
        # x represents the regression coefficients
        y_pred = self.X @ x
        mse = np.mean((self.y - y_pred) ** 2)
        return mse


def test_spsa_2_on_multicollinearity_regression():
    problem = MulticollinearityRegression(n_features=5)
    solution = spsa_2(problem, n_iter=2000, a=1e-6)

    # Check that the final objective function value is lower than the initial one
    loss_x0 = problem(problem.x0)
    loss_sol = problem(solution)
    assert loss_sol < loss_x0