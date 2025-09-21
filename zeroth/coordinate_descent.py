import numpy as np

def coordinate_descent(f, x0, n_iter=100, step_size=0.1):
    """
    Performs unconstrained optimization using the Coordinate Descent algorithm.

    This algorithm minimizes a function by iteratively performing approximate
    minimization along coordinate directions. At each iteration, it minimizes
    the function with respect to a single coordinate, keeping the others fixed.

    A real-world example of its use is in solving large-scale linear regression
    problems (e.g., Lasso) where the number of features is very large.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform over all coordinates.
        step_size (float): The size of the step to take in each coordinate direction.

    Returns:
        np.ndarray: The parameters that minimize the objective function.
    """
    x = np.copy(x0)
    n_dims = len(x)

    for _ in range(n_iter):
        for i in range(n_dims):
            # Create vectors for positive and negative steps
            step_positive = np.zeros(n_dims)
            step_positive[i] = step_size
            step_negative = np.zeros(n_dims)
            step_negative[i] = -step_size

            # Evaluate function at current point and after taking steps
            fx = f(x)
            fx_positive = f(x + step_positive)
            fx_negative = f(x + step_negative)

            # Choose the best of the three points
            if fx_positive < fx:
                x += step_positive
            elif fx_negative < fx:
                x += step_negative

    return x
