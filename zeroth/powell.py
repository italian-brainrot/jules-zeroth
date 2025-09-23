import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp, trapezoid

from zeroth.problem import Problem


def powell(f: Problem, n_iter: int = 10, tol: float = 1e-5):
    """
    Performs unconstrained optimization using Powell's conjugate direction method.

    Args:
        f (Problem): The objective function to minimize.
        n_iter (int): The number of iterations to perform.
        tol (float): The tolerance for convergence.

    Returns:
        np.ndarray: The best solution found.
    """
    n = f.ndim
    x = f.x0.copy()
    dirs = np.eye(n)
    fx = f(x)

    for _ in range(n_iter):
        p_start = np.copy(x)
        fx_start = fx

        for i in range(n):
            def line_search_func(alpha):
                return f(x + alpha * dirs[i])

            res = minimize_scalar(line_search_func)
            x += res.x * dirs[i]

        new_dir = x - p_start
        x_new = x + new_dir
        fx_new = f(x_new)

        if fx_new < fx_start:
            # Replace the direction of largest decrease with the new direction
            magnitudes = np.array([f(x + d) - fx for d in dirs])
            idx_max = np.argmax(magnitudes)
            dirs[idx_max] = new_dir
            x = x_new
            fx = fx_new

        if np.linalg.norm(x - p_start) < tol:
            break

    return x


class CSTROptimization(Problem):
    """
    This problem is to find the optimal control input `u` for a
    continuously stirred-tank reactor (CSTR) to minimize a
    performance measure. The problem is from Example 6.4-3 of
    "Optimal Control Theory: An Introduction" by Donald E. Kirk.

    The control input `u(t)` is a piecewise constant function.
    """
    def __init__(self, n_dims=10):
        super().__init__(x0=np.zeros(n_dims))
        self.n_dims = n_dims

    def evaluate(self, x):
        total_time = 0.78
        time_step = total_time / self.n_dims
        y_current = np.array([0.5, 0.5])
        total_J = 0.0

        for i in range(self.n_dims):
            u_current = x[i]
            t_start = i * time_step
            t_end = (i + 1) * time_step
            t_span = [t_start, t_end]

            def cstr_dynamics(t, y):
                x1, x2 = y
                dx1_dt = -x1 + 0.5 * x2 + (1 - np.exp(0.5 * x1)) * u_current
                dx2_dt = -x2 - (1 - np.exp(0.5 * x1)) * u_current
                return [dx1_dt, dx2_dt]

            sol = solve_ivp(cstr_dynamics, t_span, y_current)

            y_current = sol.y[:, -1]

            integrand = sol.y[0]**2 + sol.y[1]**2 + 0.1 * u_current**2
            J_i = trapezoid(integrand, sol.t)
            total_J += J_i

        return total_J


def test_powell_on_cstr():
    problem = CSTROptimization(n_dims=10)
    solution = powell(problem, n_iter=20)

    initial_cost = problem(problem.x0)
    final_cost = problem(solution)

    assert final_cost < initial_cost
