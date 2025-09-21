# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides short reference implementations of unconstrained zeroth-order optimization algorithms using NumPy.

The repository follows a simple consistent structure. Each file contains three components:
1. A short algorithm implementation following a simple API: `algorithm(f, x0, ...)`
2. A short reference implementation of a unique real-world problem that this algorithm is commonly used for, following the `Problem` class API.
3. A test case that verifies the algorithm's implementation on the reference problem.
