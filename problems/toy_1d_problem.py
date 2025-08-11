from optprob.problems import ConstrainedScalarOptimizationProblem
import numpy as np
import scipy


class Toy1DProblem(ConstrainedScalarOptimizationProblem):

    def __init__(self):
        bounds = [(-5.0, 5.0)]
        name = "Toy1DProblem"
        super().__init__(bounds, name=name, global_minimum=[2.5085382557867626])

    @staticmethod
    def cost_function_to_minimize(x) -> float:
        return 1.0 / (-0.05 * x[0] ** 2 - np.cos(x[0]) + 0.25 * np.sin(3 * x[0] + 0.8) + 5)


# Test problem instance
problem = Toy1DProblem()
assert str(problem) == (
    "Toy1DProblem(_name='Toy1DProblem', _guesses=[], _input_dim=1, "
    "_bounds=[(-5.0, 5.0)], _global_minimum=[2.5085382557867626])"
)
assert problem.bounds == [(-5.,  5.)]
assert problem.input_dim == 1
assert problem.nfev == 0
assert problem.guesses == []
assert problem([0.5]) == 0.23275605031813504
assert problem.guesses == [(0.23275605031813504, [0.5])]
assert problem.nfev == 1
assert problem([-5]) == 0.3108649328945798
assert problem([5]) == 0.29041392127738885
assert problem.nfev == 3
assert problem.best_guess == (0.23275605031813504, [0.5])

# Find global minimum using Scipy and a good initial guess
sol = scipy.optimize.minimize(problem, x0=2.5, bounds=problem.bounds, tol=1e-15)
assert sol.status == 0
print(sol.fun, sol.x.item())
assert np.array_equal(problem.global_minimum, sol.x)
