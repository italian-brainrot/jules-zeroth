import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp, trapezoid

from zeroth.problem import Problem


def powell(f: Problem, x0: np.ndarray, n_iter: int = 10, tol: float = 1e-5):
    """
    Performs unconstrained optimization using Powell's conjugate direction method.

    Args:
        f (Problem): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform.
        tol (float): The tolerance for convergence.

    Returns:
        np.ndarray: The best solution found.
    """
    n = len(x0)
    x = np.copy(x0)
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

    The control input `u(t)` is a piecewise constant function with two steps.
    """
    def __init__(self):
        super().__init__(x0=np.array([0.0, 0.0]))

    def evaluate(self, x):
        u1, u2 = x

        def cstr_dynamics_u1(t, y):
            x1, x2 = y
            u = u1
            dx1_dt = -x1 + 0.5 * x2 + (1 - np.exp(0.5 * x1)) * u
            dx2_dt = -x2 - (1 - np.exp(0.5 * x1)) * u
            return [dx1_dt, dx2_dt]

        def cstr_dynamics_u2(t, y):
            x1, x2 = y
            u = u2
            dx1_dt = -x1 + 0.5 * x2 + (1 - np.exp(0.5 * x1)) * u
            dx2_dt = -x2 - (1 - np.exp(0.5 * x1)) * u
            return [dx1_dt, dx2_dt]

        t_span1 = [0, 0.39]
        y0 = [0.5, 0.5]
        sol1 = solve_ivp(cstr_dynamics_u1, t_span1, y0)

        y1_end = sol1.y[:, -1]
        t_span2 = [0.39, 0.78]
        sol2 = solve_ivp(cstr_dynamics_u2, t_span2, y1_end)

        integrand1 = sol1.y[0]**2 + sol1.y[1]**2 + 0.1 * u1**2
        J1 = trapezoid(integrand1, sol1.t)

        integrand2 = sol2.y[0]**2 + sol2.y[1]**2 + 0.1 * u2**2
        J2 = trapezoid(integrand2, sol2.t)

        return J1 + J2


def test_powell_on_cstr():
    problem = CSTROptimization()
    solution = powell(problem, problem.x0)

    initial_cost = problem(problem.x0)
    final_cost = problem(solution)

    assert final_cost < initial_cost
