# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides short reference implementations of unconstrained zeroth-order optimization algorithms using NumPy.

The repository follows a simple consistent structure. To add an algorithm, we do this:
1. Create a clear and consize algorithm implementation following a simple API: `algorithm(f, x0, ...)`
2. Find small unique real-world problem that this algorithm has been used for in literature, and create a clear and consize implementation following the `Problem` class API.
3. Add a test case that verifies the algorithm's implementation on the reference problem.

For simplicity each .py file contains algorithm, problem and a test case.
