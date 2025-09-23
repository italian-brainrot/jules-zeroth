# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides reference implementations of unconstrained and bound constrained zeroth-order optimization algorithms with focus on correctness using NumPy.

The repository follows a simple consistent structure. To add an algorithm, we do this:

1. Create a clear and concise algorithm implementation following a simple API: `algorithm(f: Problem, x0: np.ndarray, ...)`. If the problem is unbounded, algorithms assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). Some algorithms arer able to find solutions far from this initial prior. Algorithms that support bounds assume them to be (-1, 1) for all variables.

2. Find a small unique real-world problem that this algorithm has been used for in literature, and create a clear and concise implementation subclassing the `Problem` class. The `Problem` class takes care of scaling bounded problems to (-1, 1) range. If problem is unbounded and has very different parameter scales, it should be parameterized to have approximately centered and reduced prior.

3. Add a test case that verifies the algorithm's implementation on the reference problem.

For simplicity each .py file contains algorithm, problem and a test case. All algorithms can be used on all problems, since they follow the same API.

## Requirements

- `numpy`
- `pytest` (to run tests)
