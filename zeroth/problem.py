import numpy as np

def _get_bound(b: float | np.ndarray | None, x0: np.ndarray):
    if b is None:
        return None
    return np.broadcast_to(b, x0.shape)

class Problem:
    def __init__(self, x0: np.ndarray, lb: float | np.ndarray | None = None, ub: float | np.ndarray | None = None):

        # verify the shape
        if x0.ndim != 1:
            raise ValueError(f"x0 must be a vector, got {x0.shape = }")

        self.x0 = x0

        # bounds
        if len([i for i in (lb,ub) if i is not None]) == 1:
            raise ValueError("either both `lb`, `ub` should be None, or both should be specified")

        self.lb = _get_bound(lb, x0)
        """lower bounds array of same shape as x0"""

        self.ub = _get_bound(ub, x0)
        """upper bounds array of same shape as x0"""

    def __call__(self, x):
        """The objective function to minimize."""
        return self.f(x)
