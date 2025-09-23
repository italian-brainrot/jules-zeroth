from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

def _get_bound(b: float | ArrayLike | None, shape: tuple) -> np.ndarray | None:
    if b is None:
        return None
    return np.broadcast_to(b, shape)

def _normalize(x: np.ndarray, lb: np.ndarray | None, ub: np.ndarray | None):
    if lb is not None and ub is not None:
        x = (x - lb) / (ub - lb)
        x = (x * 2) - 1
    return x

def _unnormalize(x: np.ndarray, lb: np.ndarray | None, ub: np.ndarray | None):
    if lb is not None and ub is not None:
        x = (x + 1) / 2
        x = x * (ub - lb) + lb
    return x


class Problem(ABC):
    def __init__(self, x0: ArrayLike, lb: float | ArrayLike | None = None, ub: float | ArrayLike | None = None):
        """Initialize a problem.

        Args:
            x0 (ArrayLike):
                Initial guess. Array of real elements of size (n,), where n is the number of variables.
            lb (float | ArrayLike | None, optional):
                lower bounds - integer, array of size (n, ), or None for unbounded problems. Defaults to None.
            ub (float | ArrayLike | None, optional):
                upper bounds - integer, array of size (n, ), or None for unbounded problems. Defaults to None.

        Keep problems unbounded whenever possible. If a problem is unbounded, make sure to approximately scale the variables as optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.

        If problem is bounded, search space is automatically normalized to be in (-1, 1) range.
        """

        x0 = np.array(x0)
        if x0.ndim != 1:
            raise ValueError(f"x0 must be a vector, got {x0.shape = }")

        if len([i for i in (lb,ub) if i is not None]) == 1:
            raise ValueError("either both `lb`, `ub` should be None, or both should be specified")

        self.ndim: int = x0.size
        """number of variables."""

        self._lb: np.ndarray | None = _get_bound(lb, shape=(self.ndim,))
        """Lower bounds array of same shape as x0, or None if problem is unbounded."""

        self._ub: np.ndarray | None = _get_bound(ub, shape=(self.ndim,))
        """Upper bounds array of same shape as x0, or None if problem is unbounded."""

        self.bounded: bool = self._lb is not None and self._ub is not None
        """True if problem is bounded, bounds are normalized to be (-1, 1)"""

        self.x0: np.ndarray = _normalize(x0, self._lb, self._ub)
        """initial guess."""

    @abstractmethod
    def evaluate(self, x: np.ndarray):
        """Evaluates the objective function at ``x``."""
        raise NotImplementedError(f"{self.__class__.__name__} needs to implement `__call__`")

    def __call__(self, x: np.ndarray):
        penalty = 0

        # if problem is bounded, search space is normalized to be in (-1, 1) range.
        # clip and add penalty for unbounded algorithms, then un-normalize back to original search domain
        if self._lb is not None or self._ub is not None:

            x_clipped = x.clip(-1, 1)
            penalty = np.mean((x - x_clipped) ** 2)
            x = _unnormalize(x_clipped, self._lb, self._ub)

        return self.evaluate(x) + penalty