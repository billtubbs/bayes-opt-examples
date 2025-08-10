import numpy as np
from dataclasses import dataclass
from collections import Counter


@dataclass
class ConstrainedScalarOptimizationProblem():
    _name: str
    _guesses: list
    _input_dim: int
    _bounds: np.ndarray
    _global_minimum: np.ndarray

    def __init__(self, bounds, name=None, global_minimum=None):
        self._bounds = bounds
        self._input_dim = len(bounds)
        self._name = name
        self._global_minimum = global_minimum
        self.reset()

    def reset(self):
        self._guesses = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_dim(self) -> tuple:
        return self._input_dim

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def global_minimum(self) -> np.ndarray:
        return self._global_minimum

    @property
    def nfev(self) -> int:
        return len(self._guesses)

    @property
    def guesses(self) -> list:
        return self._guesses

    @property
    def best_guess(self) -> tuple:
        return min(self._guesses)

    @staticmethod
    def cost_function_to_minimize(x, *args) -> float:
        # Implement cost function to minimize here
        cost = 0.0
        return cost

    def __call__(self, x, *args) -> float:
        assert np.all(
            (b[0] <= xi) and (xi <= b[1]) for b, xi in zip(self._bounds, x)
        )
        cost = self.cost_function_to_minimize(x, *args)
        self._guesses.append((cost, x))
        return cost


def solve_problem_with_optimizer(problem, minimizer, *args, **kwargs):
    problem.reset()
    sol = minimizer(problem, *args, **kwargs)
    return sol


def solve_problem_with_optimizer_n_repeats(
    problem, minimizer, n_repeats, *args, decimals=6, **kwargs
):

    solutions = []
    fun_evals = []
    for i in range(n_repeats):
        problem.reset()
        sol = minimizer(problem, *args, **kwargs)
        solutions.append(tuple(round(float(xi), decimals) for xi in sol.x))
        fun_evals.append(np.array([f[0] for f in problem.guesses]))
    unique_solutions = Counter(solutions)
    return fun_evals, unique_solutions
