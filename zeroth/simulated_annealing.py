import numpy as np

from zeroth.problem import Problem


def simulated_annealing(
    f: Problem,
    n_iter=1000,
    temp=1.0,
    cooling_rate=0.99,
    discrete=False,
):
    """
    Performs optimization using the Simulated Annealing algorithm.

    This algorithm explores the search space by accepting worse solutions with a
    probability that decreases over time.

    Args:
        f (Problem): The objective function to minimize.
        n_iter (int): The number of iterations to perform.
        temp (float): The initial temperature.
        cooling_rate (float): The rate at which the temperature cools.
        discrete (bool): Whether the problem is discrete.

    Returns:
        np.ndarray: The best solution found.
    """
    current_x = f.x0
    current_fx = f(current_x)

    best_x = current_x
    best_fx = current_fx

    for _ in range(n_iter):
        # Generate a random neighbor
        if discrete:
            neighbor_x = current_x.copy()
            idx_to_flip = np.random.randint(0, f.ndim)
            neighbor_x[idx_to_flip] = 1 - neighbor_x[idx_to_flip]
        elif f.bounded:
            neighbor_x = current_x + np.random.uniform(-0.1, 0.1, f.ndim)
            neighbor_x = np.clip(neighbor_x, -1, 1)
        else:
            neighbor_x = current_x + np.random.randn(f.ndim) * 0.1

        neighbor_fx = f(neighbor_x)

        # Decide whether to move to the neighbor
        if neighbor_fx < current_fx:
            current_x = neighbor_x
            current_fx = neighbor_fx
        else:
            delta = neighbor_fx - current_fx
            if np.random.rand() < np.exp(-delta / temp):
                current_x = neighbor_x
                current_fx = neighbor_fx

        # Update the best solution found so far
        if current_fx < best_fx:
            best_x = current_x
            best_fx = current_fx

        # Cool the temperature
        temp *= cooling_rate

    return best_x


class UncapacitatedFacilityLocation(Problem):
    """
    A real-world problem of locating facilities to serve customers.

    The goal is to decide where to open facilities to minimize the sum of
    facility opening costs and customer transportation costs.
    """
    def __init__(self, n_customers=10, n_facilities=5, seed=0):
        np.random.seed(seed)
        self.n_customers = n_customers
        self.n_facilities = n_facilities

        # Customer locations
        self.customer_locs = np.random.rand(n_customers, 2)

        # Facility locations and opening costs
        self.facility_locs = np.random.rand(n_facilities, 2)
        self.opening_costs = np.random.randint(10, 30, n_facilities)

        # Transportation costs (proportional to distance)
        self.transport_costs = np.linalg.norm(
            self.customer_locs[:, np.newaxis, :] - self.facility_locs[np.newaxis, :, :],
            axis=2
        )

        # Initial guess: open all facilities
        x0 = np.ones(n_facilities)
        super().__init__(x0)

    def evaluate(self, x):
        # x is a binary vector indicating which facilities are open
        open_facilities = np.where(x > 0.5)[0]

        if len(open_facilities) == 0:
            return np.inf

        # Cost of opening facilities
        total_cost = np.sum(self.opening_costs[open_facilities])

        # Cost of serving customers
        # Each customer is served by the nearest open facility
        min_transport_costs = np.min(self.transport_costs[:, open_facilities], axis=1)
        total_cost += np.sum(min_transport_costs)

        return total_cost


def test_simulated_annealing_on_uflp():
    problem = UncapacitatedFacilityLocation(n_customers=20, n_facilities=10, seed=0)
    solution = simulated_annealing(problem, n_iter=2000, discrete=True)

    # Check that the final objective function value is lower than the initial one
    loss_x0 = problem(problem.x0)
    loss_sol = problem(solution)
    assert loss_sol < loss_x0
