import numpy as np

from zeroth.problem import Problem


def nelder_mead(f: Problem, x0=None, max_iter=1000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Minimizes a function using the Nelder-Mead algorithm.

    This algorithm is a direct search method that uses a simplex of n+1 points
    for an n-dimensional search space. The simplex adapts to the local landscape
    by reflecting, expanding, contracting, and shrinking.

    Args:
        f (Problem): The objective function to minimize.
        x0 (np.ndarray, optional): Initial guess. Defaults to problem's x0.
        max_iter (int): The maximum number of iterations.
        alpha (float): The reflection coefficient.
        gamma (float): The expansion coefficient.
        rho (float): The contraction coefficient.
        sigma (float): The shrink coefficient.

    Returns:
        np.ndarray: The best solution found.
    """
    if x0 is None:
        x0 = f.x0
    n = len(x0)

    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        point = x0.copy()
        point[i] = x0[i] + 1.0  # Initial step
        simplex.append(point)

    for i in range(max_iter):
        # Order simplex by function values
        scores = [(f(point), point) for point in simplex]
        scores.sort(key=lambda x: x[0])
        simplex = [s[1] for s in scores]

        best_point = simplex[0]
        worst_point = simplex[-1]
        second_worst_point = simplex[-2]

        # Centroid of the simplex except for the worst point
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - worst_point)
        fxr = f(xr)

        if scores[0][0] <= fxr < scores[-2][0]:
            simplex[-1] = xr
            continue

        # Expansion
        if fxr < scores[0][0]:
            xe = centroid + gamma * (xr - centroid)
            fxe = f(xe)
            if fxe < fxr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
            continue

        # Contraction
        xc = centroid + rho * (worst_point - centroid)
        fxc = f(xc)
        if fxc < f(worst_point):
            simplex[-1] = xc
            continue

        # Shrink
        for j in range(1, len(simplex)):
            simplex[j] = best_point + sigma * (simplex[j] - best_point)

    return simplex[0]


class WeldedBeamDesign(Problem):
    """
    The Welded Beam Design optimization problem.

    This is a classic structural optimization problem where the goal is to
    minimize the cost of a welded beam subject to several constraints.
    The design variables are the dimensions of the weld and the beam.

    Variables:
    - h (x1): weld thickness
    - l (x2): weld length
    - t (x3): beam thickness
    - b (x4): beam width

    Constraints:
    - Shear stress in the weld (τ)
    - Bending stress in the beam (σ)
    - Buckling load on the bar (Pc)
    - End deflection of the beam (δ)
    - Side constraints
    """
    def __init__(self):
        # [h, l, t, b]
        x0 = np.array([0.2, 3.0, 8.0, 0.2])
        lb = np.array([0.125, 0.1, 0.1, 0.125])
        ub = np.array([5.0, 10.0, 10.0, 5.0])
        super().__init__(x0, lb=lb, ub=ub)

    def evaluate(self, x):
        h, l, t, b = x[0], x[1], x[2], x[3]

        # Objective function
        cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

        # Constraints
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        tau_max = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25

        M = P * (L + l / 2.0)
        R = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0)**2)
        J = 2 * (np.sqrt(2) * h * l * (l**2 / 12.0 + ((h + t) / 2.0)**2))

        tau_prime = P / (np.sqrt(2) * h * l)
        tau_double_prime = (M * R) / J
        tau = np.sqrt(tau_prime**2 + 2 * tau_prime * tau_double_prime * (l / (2 * R)) + tau_double_prime**2)

        sigma = (6 * P * L) / (b * t**2)

        delta = (4 * P * L**3) / (E * b * t**3)

        Pc = (4.013 * E * np.sqrt((t**2 * b**6) / 36.0)) / L**2 * (1 - (t / (2 * L)) * np.sqrt(E / (4 * G)))

        # Penalty for constraint violations
        penalty = 0
        if tau > tau_max:
            penalty += (tau - tau_max)**2
        if sigma > sigma_max:
            penalty += (sigma - sigma_max)**2
        if delta > delta_max:
            penalty += (delta - delta_max)**2
        if h > b:
            penalty += (h - b)**2
        if P > Pc:
            penalty += (P - Pc)**2

        return cost + penalty * 1e9


def test_nelder_mead_on_welded_beam():
    """
    Tests the Nelder-Mead algorithm on the Welded Beam Design problem.
    """
    problem = WeldedBeamDesign()
    initial_cost = problem(problem.x0)
    solution = nelder_mead(problem)
    final_cost = problem(solution)
    assert final_cost < initial_cost
