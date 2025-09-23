import numpy as np

def _get_bound(b: float | np.ndarray | None, shape: tuple):
    if b is None:
        return None
    return np.broadcast_to(b, shape)

class Problem:
    def __init__(self, x0: np.ndarray | None = None, n_dims: int | None = None, lb: float | np.ndarray | None = None, ub: float | np.ndarray | None = None):
        if x0 is None and n_dims is None:
            raise ValueError("Either x0 or n_dims must be provided.")

        if x0 is not None:
            if x0.ndim != 1:
                raise ValueError(f"x0 must be a vector, got {x0.shape = }")
            self.x0 = x0
            self.n_dims = len(x0)
        else:
            self.x0 = None
            self.n_dims = n_dims

        if len([i for i in (lb,ub) if i is not None]) == 1:
            raise ValueError("either both `lb`, `ub` should be None, or both should be specified")

        shape = (self.n_dims,)
        self.lb = _get_bound(lb, shape)
        """lower bounds array of same shape as x0"""

        self.ub = _get_bound(ub, shape)
        """upper bounds array of same shape as x0"""

    def __call__(self, x: np.ndarray):
        """The objective function to minimize."""
        raise NotImplementedError(f"{self.__class__.__name__} needs to implement `__call__`")
