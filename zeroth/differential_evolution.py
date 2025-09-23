import numpy as np

from zeroth.problem import Problem


def differential_evolution(f: Problem, pop_size=50, mutation_factor=0.8, crossover_rate=0.7, max_iter=100):
    """
    Performs optimization using the Differential Evolution algorithm.

    Differential Evolution (DE) is a stochastic population-based optimization
    algorithm that works by iteratively improving a population of candidate
    solutions. It is well-suited for global optimization problems with
    continuous variables.

    A real-world example of its use is in tuning the parameters of a controller
    for a robotic arm to minimize tracking error.

    Args:
        f (function): The objective function to minimize.
        pop_size (int): The number of individuals in the population.
        mutation_factor (float): The mutation factor (F).
        crossover_rate (float): The crossover rate (CR).
        max_iter (int): The maximum number of generations.

    Returns:
        np.ndarray: The best solution found.
    """
    n_dims = f.n_dims
    lower_bounds = f.lb
    upper_bounds = f.ub

    # Initialize population
    population = lower_bounds + np.random.rand(pop_size, n_dims) * (upper_bounds - lower_bounds)

    for _ in range(max_iter):
        for i in range(pop_size):
            # Choose three distinct individuals
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            # Mutation
            mutant = a + mutation_factor * (b - c)
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # Crossover
            cross_points = np.random.rand(n_dims) < crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dims)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Selection
            if f(trial) < f(population[i]):
                population[i] = trial

    # Find the best individual
    best_idx = np.argmin([f(ind) for ind in population])
    return population[best_idx]


class FMSoundWaveMatching(Problem):
    """
    A problem of matching a target frequency-modulated (FM) sound wave.

    The goal is to find the parameters of an FM synthesizer that generate a
    sound wave matching a target wave. The FM wave is defined as:
    y(t) = a * sin(2 * pi * f_c * t + I * sin(2 * pi * f_m * t))

    The parameters to optimize are:
    - a: amplitude
    - f_c: carrier frequency
    - I: modulation index
    - f_m: modulation frequency
    """
    def __init__(self):
        # Target parameters
        self.target_params = np.array([1.0, 220.0, 5.0, 110.0])
        self.bounds = [(0, 2), (100, 400), (0, 10), (50, 200)]
        self.time = np.linspace(0, 0.1, 1000)
        self.target_wave = self._generate_wave(self.target_params)

        super().__init__(n_dims=4, lb=[b[0] for b in self.bounds], ub=[b[1] for b in self.bounds])

    def _generate_wave(self, params):
        a, f_c, I, f_m = params
        return a * np.sin(2 * np.pi * f_c * self.time + I * np.sin(2 * np.pi * f_m * self.time))

    def __call__(self, x):
        generated_wave = self._generate_wave(x)
        return np.sum((generated_wave - self.target_wave)**2)


def test_differential_evolution_on_fm_sound_wave():
    """
    Tests the Differential Evolution algorithm on the FM sound wave matching problem.
    """
    problem = FMSoundWaveMatching()

    # A random solution should have a high cost
    random_solution = np.array([
        np.random.uniform(b[0], b[1]) for b in problem.bounds
    ])
    initial_cost = problem(random_solution)

    # Run the differential evolution algorithm
    solution = differential_evolution(problem, pop_size=50, max_iter=200)

    # Check that the found solution is better than the random one
    final_cost = problem(solution)
    assert final_cost < initial_cost
