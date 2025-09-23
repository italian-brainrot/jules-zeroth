import numpy as np

from zeroth.problem import Problem


def stochastic_hill_climbing(f: Problem, x0, n_iter=1000, step_size=0.1):
    """
    Performs unconstrained optimization using the Stochastic Hill Climbing algorithm.

    This algorithm explores the search space by generating random steps
    from the current best point. If a new point yields a better objective
    function value, it becomes the new best point.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform.
        step_size (float): The standard deviation of the random steps.

    Returns:
        np.ndarray: The best solution found.
    """
    best_x = x0
    best_fx = f(best_x)

    for _ in range(n_iter):
        x = best_x + np.random.normal(scale=step_size, size=x0.shape)
        fx = f(x)

        if fx < best_fx:
            best_x = x
            best_fx = fx

    return best_x


class BeamDesignProblem(Problem):
    """
    A real-world problem of designing a beam to minimize its weight while
    satisfying stress constraints.
    """
    def __init__(self, L=2.0, P=5000, S=250e6):
        # Problem constants
        self.L = L  # meters
        self.P = P  # Newtons
        self.S = S  # Pascals (yield strength of steel)
        self.M = (self.P * self.L) / 4  # Maximum bending moment

        # Initial guess for width and height (w, h)
        x0 = np.array([0.1, 0.1])
        super().__init__(x0, lb=0.01, ub=1)

    def evaluate(self, x):
        """
        Calculates the cross-sectional area of the beam, with a penalty
        for violating the stress constraint.
        """
        w, h = x
        area = w * h

        # Calculate stress
        I = (w * h**3) / 12
        y = h / 2
        sigma = (self.M * y) / I

        # Penalty for violating stress constraint
        penalty = 0
        if sigma > self.S:
            penalty = 1e9 * (sigma - self.S)

        return area + penalty


def test_stochastic_hill_climbing_on_beam_design():
    problem = BeamDesignProblem()
    best_params = stochastic_hill_climbing(problem, problem.x0, n_iter=1000, step_size=0.01)

    initial_obj = problem(problem.x0)
    final_obj = problem(best_params)

    assert final_obj < initial_obj
