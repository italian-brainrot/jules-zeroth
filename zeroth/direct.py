import numpy as np

from zeroth.problem import Problem


def direct(f: Problem, max_iter=100, max_evals=1000):
    """
    DIviding RECTangles (DIRECT) algorithm for global optimization.

    DIRECT is a deterministic, bound-constrained global optimization algorithm.
    It works by systematically dividing the search space into smaller
    hyper-rectangles, balancing global exploration and local exploitation.

    This implementation is based on the description by D. R. Jones et al.

    Args:
        f (Problem): The objective function to minimize. Must be a bounded problem.
        max_iter (int): The maximum number of iterations.
        max_evals (int): The maximum number of function evaluations.

    Returns:
        np.ndarray: The best solution found in the normalized space.
    """
    if not f.bounded:
        raise ValueError("DIRECT algorithm requires a bounded problem.")

    n = f.ndim
    # Start at the problem's initial guess instead of the domain center
    # to avoid pathological points like all atoms overlapping.
    initial_center = f.x0

    # List to store rectangles, each as a dict
    # The first rectangle is still the entire domain.
    rectangles = [{
        "center": initial_center,
        "f_val": f(initial_center),
        "sides": np.ones(n)  # half-lengths of sides
    }]

    num_evals = 1
    best_f_val = rectangles[0]["f_val"]
    best_x = initial_center.copy()

    for k in range(max_iter):
        if num_evals >= max_evals:
            break

        # 1. Identify potentially optimal rectangles
        potentially_optimal = []
        # Separate rectangles by their diagonal lengths (size)
        # Using L2 norm of side lengths as a measure of size
        sizes = [np.linalg.norm(r["sides"]) for r in rectangles]

        # Find the convex hull on the lower right
        # A rectangle 'i' is potentially optimal if no other rectangle 'j'
        # is both smaller (sizes[j] < sizes[i]) and has a better value (f[j] < f[i])
        for i in range(len(rectangles)):
            is_optimal = True
            for j in range(len(rectangles)):
                if i == j:
                    continue
                if sizes[j] < sizes[i] and rectangles[j]["f_val"] < rectangles[i]["f_val"]:
                    is_optimal = False
                    break
            if is_optimal:
                potentially_optimal.append(i)

        # 2. Divide potentially optimal rectangles
        for rect_idx in potentially_optimal:
            rect = rectangles[rect_idx]

            # Find dimensions with the longest side
            longest_sides = np.where(rect["sides"] == np.max(rect["sides"]))[0]

            # Trisect and sample along these dimensions
            delta = rect["sides"][longest_sides[0]] / 3.0

            for dim in longest_sides:
                c_plus = rect["center"].copy()
                c_minus = rect["center"].copy()

                c_plus[dim] += delta
                c_minus[dim] -= delta

                f_plus = f(c_plus)
                f_minus = f(c_minus)
                num_evals += 2

                if f_plus < best_f_val:
                    best_f_val = f_plus
                    best_x = c_plus
                if f_minus < best_f_val:
                    best_f_val = f_minus
                    best_x = c_minus

                # Create new smaller rectangles
                new_sides = rect["sides"].copy()
                new_sides[dim] /= 3.0

                rectangles.append({"center": c_plus, "f_val": f_plus, "sides": new_sides})
                rectangles.append({"center": c_minus, "f_val": f_minus, "sides": new_sides})

                if num_evals >= max_evals:
                    break

            # Update the divided rectangle's size
            rect["sides"][longest_sides] /= 3.0

            if num_evals >= max_evals:
                break

    return best_x


class LennardJonesPotential(Problem):
    """
    The Lennard-Jones potential problem for finding the minimum energy
    configuration of a cluster of N atoms in 2D.

    The potential is a function of the distances between atoms and is known
    for having a large number of local minima, making it a classic benchmark
    for global optimization algorithms.
    """
    def __init__(self, n_atoms=4):
        self.n_atoms = n_atoms
        self.epsilon = 1.0  # Depth of the potential well
        self.sigma = 1.0    # Finite distance at which inter-particle potential is zero

        ndim = n_atoms * 2  # 2D coordinates for each atom

        # Initial positions in a rough grid
        np.random.seed(42)
        x0 = np.random.rand(ndim) * 2 - 1

        # Atoms are constrained to a box
        super().__init__(x0)

    def evaluate(self, x):
        """
        Calculates the total Lennard-Jones potential for a given configuration.
        The input x is a flattened array of 2D coordinates, e.g., [x1, y1, x2, y2, ...].
        """
        # Reshape the flat vector into a (n_atoms, 2) array of coordinates
        coords = x.reshape((self.n_atoms, 2)) * 2

        total_energy = 0.0
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                # Calculate the squared distance between atom i and atom j
                dist_sq = np.sum((coords[i] - coords[j])**2)

                # Avoid division by zero if atoms are on top of each other
                if dist_sq < 1e-12:
                    return 1e12 # Return a large energy value as a penalty

                inv_dist_sq = (self.sigma**2) / dist_sq
                inv_dist_6 = inv_dist_sq**3
                inv_dist_12 = inv_dist_6**2

                total_energy += 4.0 * self.epsilon * (inv_dist_12 - inv_dist_6)

        return total_energy


def test_direct_on_lennard_jones():
    """
    Tests the DIRECT algorithm on the Lennard-Jones potential problem.
    """
    # Use a small number of atoms for a quick test
    problem = LennardJonesPotential(n_atoms=4)

    # A random initial configuration should have a relatively high energy
    initial_energy = problem(problem.x0)

    # Run the DIRECT algorithm to find a better configuration
    # Use a small number of evaluations for the test to run quickly
    solution = direct(problem, max_iter=30, max_evals=150)

    # Check that the found solution has a lower energy
    final_energy = problem(solution)
    assert final_energy < initial_energy
