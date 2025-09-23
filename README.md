# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides short reference implementations of unconstrained or bound constrained zeroth-order optimization algorithms using NumPy.

The repository follows a simple consistent structure. To add an algorithm, we do this:

1. Create a clear and concise algorithm implementation following a simple API: `algorithm(f, x0, ...)`. Unbounded algorithms assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior. Bounded algorithms assume bounds to be (-1, 1) for all variables.

2. Find small unique real-world problem that this algorithm has been used for in literature, and create a clear and concise implementation subclassing the `Problem` class. Unbounded problems should be rescaled to have approximately centered and reduced prior. The `Problem` class takes care of scaling bounded problems.

3. Add a test case that verifies the algorithm's implementation on the reference problem.

For simplicity each .py file contains algorithm, problem and a test case. All algorithms can be used on all problems, since they follow the same API.

## Requirements

- `numpy`
- `pytest` (to run tests)
