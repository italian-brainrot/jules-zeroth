import numpy as np

from zeroth.problem import Problem


def spsa(f: Problem, n_iter=1000, a=0.01, c=0.01, A=100, alpha=0.602, gamma=0.101):
    """
    Minimizes a function using the Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm.

    Args:
        f (Problem): The problem to solve.
        n_iter (int): The number of iterations.
        a (float): SPSA parameter.
        c (float): SPSA parameter.
        A (float): SPSA stability constant.
        alpha (float): SPSA parameter.
        gamma (float): SPSA parameter.

    Returns:
        np.ndarray: The best solution found.
    """
    x = f.x0
    for k in range(n_iter):
        ak = a / (k + 1.0 + A) ** alpha
        ck = c / (k + 1.0) ** gamma

        # Generate a random perturbation vector
        delta = np.random.choice([-1, 1], size=f.ndim)

        # Estimate the gradient
        x_plus = x + ck * delta
        x_minus = x - ck * delta

        y_plus = f(x_plus)
        y_minus = f(x_minus)

        grad = (y_plus - y_minus) / (2 * ck * delta)

        # Update the solution
        x = x - ak * grad

        if f.bounded:
            x = np.clip(x, -1, 1)

    return x


class SignalDenoising(Problem):
    """
    A real-world problem of denoising a signal using a FIR filter.

    The goal is to find the optimal coefficients for a 3-tap FIR filter
    to remove noise from a signal. SPSA is well-suited for this kind of
    high-dimensional optimization problem where the gradient is not
    available.
    """
    def __init__(self):
        # Generate a synthetic signal
        np.random.seed(0)
        fs = 1000  # sampling frequency
        t = np.arange(0, 1, 1 / fs)
        # A clean signal with a low frequency component
        self.clean_signal = np.sin(2 * np.pi * 50 * t)
        # Add some high-frequency noise
        noise = 0.5 * np.sin(2 * np.pi * 250 * t) + 0.25 * np.random.randn(len(t))
        self.noisy_signal = self.clean_signal + noise

        # Initial guess for the filter coefficients
        x0 = np.array([0.33, 0.33, 0.33]) # a simple averaging filter
        # The problem class will scale this to (-1,1)
        super().__init__(x0))

    def evaluate(self, x):
        # x contains the 3 FIR filter coefficients
        b = x

        # Apply the FIR filter using convolution
        filtered_signal = np.convolve(self.noisy_signal, b, mode='same')

        # Calculate the mean squared error.
        mse = np.mean((self.clean_signal - filtered_signal) ** 2)
        return mse


def test_spsa_on_signal_denoising():
    problem = SignalDenoising()
    solution = spsa(problem, n_iter=1000)

    # Check that the final objective function value is lower than the initial one
    loss_x0 = problem(problem.x0)
    loss_sol = problem(solution)
    assert loss_sol < loss_x0
