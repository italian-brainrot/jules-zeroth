import numpy as np

from zeroth.problem import Problem


def cma_es(f: Problem, sigma=0.3, max_iter=100):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    CMA-ES is an evolutionary algorithm for difficult non-linear non-convex
    optimization problems. It's considered a state-of-the-art method in
    zeroth-order optimization.

    This implementation is based on the description on Wikipedia and the
    MATLAB code provided there.

    Args:
        f (Problem): The objective function to minimize.
        x0 (np.ndarray, optional): Initial guess. Defaults to problem's x0.
        sigma (float): Initial step-size.
        max_iter (int): The maximum number of generations.

    Returns:
        np.ndarray: The best solution found.
    """
    n = f.ndim
    mean = f.x0

    # Strategy parameter setting: Selection
    lambd = 4 + int(3 * np.log(n))
    mu = lambd // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = np.sum(weights)**2 / np.sum(weights**2)

    # Strategy parameter setting: Adaptation
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

    # Initialize dynamic (internal) strategy parameters and constants
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    eigeneval = 0
    chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    for g in range(max_iter):
        # Generate and evaluate lambda offspring
        arx = np.zeros((lambd, n))
        arfitness = np.zeros(lambd)
        for k in range(lambd):
            arx[k] = mean + sigma * B @ (D * np.random.randn(n))
            arfitness[k] = f(arx[k])

        # Sort by fitness
        arindex = np.argsort(arfitness)
        arfitness = arfitness[arindex]
        arx = arx[arindex]

        xold = mean
        mean = weights @ arx[:mu]

        # Cumulation: Update evolution paths
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - xold)) / sigma
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (g + 1))) / chiN < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - xold) / sigma

        # Adapt covariance matrix C
        artmp = (1 / sigma) * (arx[:mu] - xold)
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
            + cmu * (artmp.T @ np.diag(weights) @ artmp)

        # Adapt step size sigma
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # Decomposition of C into B*diag(D.^2)*B' (diagonalization)
        if g - eigeneval > lambd / (c1 + cmu) / n / 10:
            eigeneval = g
            C = np.triu(C) + np.triu(C, 1).T  # enforce symmetry
            D_diag, B = np.linalg.eigh(C)
            D = np.sqrt(D_diag)
            invsqrtC = B @ np.diag(1 / D) @ B.T

    return mean


class Rosenbrock(Problem):
    """
    The Rosenbrock function is a non-convex function used as a performance
    test problem for optimization algorithms.
    """
    def __init__(self, ndim=4):
        super().__init__(x0=np.zeros(ndim))

    def evaluate(self, x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def test_cma_es_on_rosenbrock():
    """
    Tests the CMA-ES algorithm on the Rosenbrock function.
    """
    problem = Rosenbrock(ndim=4)

    # A random solution should have a high cost
    initial_cost = problem(problem.x0)

    # Run the cma_es algorithm
    solution = cma_es(problem, max_iter=200) # Increased iterations for this difficult problem

    # Check that the found solution is better than the random one
    final_cost = problem(solution)
    assert final_cost < initial_cost
