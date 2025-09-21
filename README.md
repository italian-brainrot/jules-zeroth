# Unconstrained Zeroth-Order Optimization Algorithms

This repository provides reference implementations of unconstrained zeroth-order optimization algorithms using NumPy. These algorithms are designed to minimize an objective function `f(x)` given an initial parameter array `x0`.

## Implemented Algorithms

### 1. Random Search

- **File:** `zeroth/random_search.py`
- **Description:** This algorithm explores the search space by generating random steps from the current best point. If a new point yields a better objective function value, it becomes the new best point.
- **Real-world Example:** Hyperparameter tuning for machine learning models.

### 2. Coordinate Descent

- **File:** `zeroth/coordinate_descent.py`
- **Description:** This algorithm minimizes a function by iteratively performing approximate minimization along coordinate directions. At each iteration, it minimizes the function with respect to a single coordinate, keeping the others fixed.
- **Real-world Example:** Solving large-scale linear regression problems.

## Usage

To use an algorithm, import the corresponding function and pass your objective function and initial guess to it:

```python
import numpy as np
from zeroth.random_search import random_search

# Define a simple objective function (e.g., Sphere function)
def sphere(x):
    return np.sum(x**2)

# Set an initial guess
x0 = np.array([0.5, 0.5])

# Find the minimum
result = random_search(sphere, x0)

print(f"The minimum is at: {result}")
```
