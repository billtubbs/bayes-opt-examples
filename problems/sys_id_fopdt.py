"""
Based on the the Excel / Visual Basic implementation by Russ Rhinehart.

See:
 - Generic LF Dynamic Model Regression FOPDT 2025-03-23.xlsx

"""

import os
import numpy as np
import numba as nb
import pandas as pd
from problems.optprob.problems import ConstrainedScalarOptimizationProblem


def make_simulate_function(dt, u_data):
    n_data = u_data.shape[0]

    @nb.njit
    def simulate(
        K: float,
        tau: float,
        n_delay: int,
        y_base: float,
        u_base: float,
        y_init: float
    ) -> np.ndarray:
        alpha_comp = np.exp(-dt / tau)
        alpha = 1.0 - alpha_comp
        y_dev = y_init - y_base
        y_model = np.full((n_data, ), np.nan)
        for k in range(n_data):
            if k >= n_delay:
                # Calculate the model from the input data
                influence = u_data[k - n_delay] - u_base
            else:
                # There is no u data for negative time, so use first value, 
                # pretending all past values were the same
                influence = u_data[0] - u_base
            y_dev = alpha * K * influence + alpha_comp * y_dev
            y_model[k] = y_dev + y_base
        return y_model

    return simulate


def rms_prediction_error(y_model, y_data):
    """Root-mean-squared prediction error."""
    rms_error = np.sqrt(np.mean((y_model - y_data) ** 2))
    return rms_error


class SysIdFOPDT(ConstrainedScalarOptimizationProblem):

    def __init__(
        self,
        bounds,
        dt,
        u_data,
        y_data,
        u_base=5.0,
        name="SysIdFOPDT",
        global_minimum=(2.336465534800439, )
    ):
        super().__init__(bounds, name=name, global_minimum=global_minimum)
        self.u_data = u_data
        self.y_data = y_data
        self.simulate = make_simulate_function(dt, u_data)
        self.params = {'u_base': u_base}

    def cost_function_to_minimize(self, x, u_base=None) -> float:
        K, tau, n_delay, y_base, y_init = x
        assert isinstance(x[2], (int, np.integer))
        if u_base is None:
            u_base = self.params['u_base']
        y_model = self.simulate(K, tau, n_delay, y_base, u_base, y_init)
        rms_error = rms_prediction_error(y_model, self.y_data)
        return rms_error

    def calculate_y_model(self, x, u_base=None):
        K, tau, n_delay, y_base, y_init = x
        assert isinstance(x[2], (int, np.integer))
        if u_base is None:
            u_base = self.params['u_base']
        y_model = self.simulate(K, tau, n_delay, y_base, u_base, y_init)
        return y_model


class SysIdFOPDTRealDelay(SysIdFOPDT):

    def __init__(self, bounds, dt, u_data, y_data, u_base=5.0, name="SysIdFOPDTRealDelay"):
        super().__init__(
            bounds,
            dt,
            u_data,
            y_data,
            u_base=u_base,
            name=name,
            global_minimum=(2.336465534800439, )
        )

    def cost_function_to_minimize(self, x, u_base=None) -> float:
        K, tau, theta, y_base, y_init = x
        n_delay = int(theta / dt + 0.5)
        if u_base is None:
            u_base = self.params['u_base']
        y_model = self.simulate(K, tau, n_delay, y_base, u_base, y_init)
        rms_error = rms_prediction_error(y_model, self.y_data)
        return rms_error


def calculate_reasonable_bounds(t, u_data, y_data):

    # Set parameter bounds based on data characteristics
    y_mean = np.mean(y_data)
    y_range = np.max(y_data) - np.min(y_data)
    u_range = np.max(u_data) - np.min(u_data)

    time_span = np.max(t) - np.min(t)

    # Intelligent parameter bounds
    # Make sure these are in the same order as elements of x
    bounds = {
        'K': (-5 * y_range / u_range, 5 * y_range / u_range),    # Process gain
        'tau': (dt, time_span / 3),                              # Time constant
        'n_delay': (0, int(round(time_span / 10 / dt))),         # Time delays
        'y_base': (y_mean - 2 * y_range, y_mean + 2 * y_range),  # Output baseline
        'y_init': (                                              # Initial output
            y_data[0] - y_range, 
            y_data[0] + y_range
        )
    }
    return bounds


# UNIT TESTS
# Load input-output data
data_dir = 'data'
filename = 'io_data_fopdt.csv'
input_output_data = pd.read_csv(os.path.join(data_dir, filename))
assert input_output_data.shape == (881, 4)
assert input_output_data.columns.tolist() == [
    'Output', 'Time', 'Input1', 'Input2'
]
assert np.array_equal(
    input_output_data.iloc[0],
    [529.4264794, 5011.92, 3.8482567, 950.433]
)
assert np.array_equal(
    input_output_data.iloc[-1],
    [474.7938764, 18231.12, 6.3663577, 936.166]
)

# Prepare input-output data
input_col = 'Input1'
t = input_output_data['Time'].to_numpy()
u_data = input_output_data[input_col].to_numpy()
y_data = input_output_data['Output'].to_numpy()

# Test data is from Russ's Excel spreadsheet
test_data = {
    "RMS": 2.33646554, 
    "ymodel[1]": 535.133420093336,
    "ymodel[2]": 534.418756507855,
    "ymodel[10]": 529.54015668416,
    "ymodel[881]": 483.232422506617,
    "Km": -12.38205772,
    "Taum": 417.6967877, 
    "Thetam": 94.32477261, 
    "y-base": 500.7177154, 
    "u-base": 5.0, 
    "y-initial": 535.8743564,
    "dt": 15.08,
    "NData": 881
}

dt = test_data["dt"]
assert test_data["NData"] == y_data.shape[0]
test_simulate = make_simulate_function(dt, u_data)
test_n_delay = int(test_data["Thetam"] / dt + 0.5)
y_model_test = test_simulate(
    test_data["Km"], 
    test_data["Taum"], 
    test_n_delay, 
    test_data["y-base"], 
    test_data["u-base"], 
    test_data["y-initial"]
)
assert np.isclose(y_model_test[0], test_data["ymodel[1]"])
assert np.isclose(y_model_test[1], test_data["ymodel[2]"])
assert np.isclose(y_model_test[9], test_data["ymodel[10]"])
assert np.isclose(y_model_test[880], test_data["ymodel[881]"])
rms_error = rms_prediction_error(y_model_test, y_data)
assert np.isclose(rms_error, test_data["RMS"])

bounds = calculate_reasonable_bounds(t, u_data, y_data)
var_names = list(bounds.keys())
bounds = list(bounds.values())
