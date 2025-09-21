import numpy as np

def random_search(f, x0, n_iter=1000, step_size=0.1):
    """
    Performs unconstrained optimization using the Random Search algorithm.

    This algorithm explores the search space by generating random steps
    from the current best point. If a new point yields a better objective
    function value, it becomes the new best point.

    A real-world example of its use is in hyperparameter tuning for
    machine learning models, where the objective function is the validation
    error and the parameters are the model's hyperparameters.

    Args:
        f (function): The objective function to minimize.
        x0 (np.ndarray): The initial guess for the parameters.
        n_iter (int): The number of iterations to perform.
        step_size (float): The standard deviation of the random steps.

    Returns:
        np.ndarray: The parameters that minimize the objective function.
    """
    best_x = x0
    best_fx = f(best_x)

    for _ in range(n_iter):
        candidate_x = best_x + np.random.normal(scale=step_size, size=x0.shape)
        candidate_fx = f(candidate_x)

        if candidate_fx < best_fx:
            best_x = candidate_x
            best_fx = candidate_fx

    return best_x
