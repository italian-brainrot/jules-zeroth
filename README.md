# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides reference implementations of unconstrained and bound constrained zeroth-order optimization algorithms with focus on correctness using NumPy.

The repository follows a simple consistent structure. To add an algorithm, we do this:

1. Create a clear and concise algorithm implementation following a simple API: `algorithm(f: Problem, ...)`. If the problem is unbounded, algorithms assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). Some algorithms are able to find solutions far from this initial prior. Algorithms that support bounds assume them to be (-1, 1) for all variables if the problem is bounded.

2. Find a small unique real-world problem that this algorithm has been used for in literature, and create a clear and concise implementation subclassing the `Problem` class. The `Problem` class takes care of scaling bounded problems to (-1, 1) range. If problem is unbounded and has very different parameter scales, it should be parameterized to have approximately centered and reduced prior.

3. Add a test case that verifies the algorithm's implementation on the reference problem.

4. Add algorithm and problem to "Algorithms and problems" section in this README.

For simplicity each .py file contains algorithm, problem and a test case. All algorithms can be used on all problems, since they follow the same API.

## Algorithms and problems
| Algorithm                | Problem                                               |
| ------------------------ | ----------------------------------------------------- |
| Random search            | Elastic Net hyperparameter tuning                     |
| Stochastic hill climbing | Beam design problem                                   |
| Coordinate descent       | Linear system                                         |
| Differential evolution   | Frequency-modulated (FM) sound wave matching          |
| CMA-ES                   | Rosenbrock                                            |
| DIRECT                   | Lennard-Jones potential problem                       |
| Nelder-Mead              | Welded Beam Design                                    |
| Powell's method          | Continuously stirred-tank reactor (CSTR) optimization |
| Simulated annealing      | Uncapacitated Facility Location Problem               |
| Bayesian optimization    | SVM hyperparameter tuning                             |
| SPSA                     | Signal Denoising                                      |
| 2-SPSA                   | Multicollinearity in linear regression                |


## Requirements

- `numpy`
- `scipy`
- `pytest` (to run tests)
- `scikit-learn` (for hyper-parameter optimization problem)
