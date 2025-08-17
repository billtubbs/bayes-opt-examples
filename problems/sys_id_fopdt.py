"""
Based on the following Visual Basic code from Russ Rhinehart.

    Ndelay=INT(thetamodel/dt + 0.5)                         ‘discretized model delay
    clambda = Exp(-dt / Taumodel)                                             ‘model coefficient
    lambda = 1 - clambda                                               ‘complement to model coefficient
    ymodeldev = Yinitial - Ybase                                  ‘initial deviation variable value for model
    SSD = 0                                                                                          ‘initialize the Sum of Squared Deviations
    For DataNumber = Ndelay + 1 To Ndata                            ‘start with first value after the delay
                    influence = udata(DataNumber - Ndelay) - Ubase                       ‘calculate influence
                    ymodeldev = lambda * Kmodel * influence + clambda * ymodeldev               ‘model
                    ymodel(DataNumber) = ymodeldev + Ybase                                  ‘convert to CV
                    SSD = SSD + (ymodel(DataNumber) - ydata(DataNumber))^2 ‘sum d^2
    Next DataNumber

Data is from the Excel spreadsheet implementation from Russ Rhinehart:
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
        for k in range(n_delay + 1, n_data):
            influence = u_data[k - n_delay] - u_base
            y_dev = alpha * K * influence + alpha_comp * y_dev
            y_model[k] = y_dev + y_base
        return y_model

    return simulate


def rms_prediction_error(y_model, y_data):
    ssd = np.mean((y_model - y_data) ** 2)
    return ssd


class SysIdFOPDT(ConstrainedScalarOptimizationProblem):

    def __init__(self, bounds, dt, u_data, y_data, u_base=5.0):
        name = "SysIdFOPDT"
        super().__init__(bounds, name=name, global_minimum=[5.043253488260278])
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
        rms_error = rms_prediction_error(
            y_model[n_delay+1:],
            self.y_data[n_delay+1:]
        )
        return rms_error

    def calculate_y_model(self, x, u_base=None):
        K, tau, n_delay, y_base, y_init = x
        assert isinstance(x[2], (int, np.integer))
        if u_base is None:
            u_base = self.params['u_base']
        y_model = self.simulate(K, tau, n_delay, y_base, u_base, y_init)
        return y_model


def calculate_reasonable_bounds(t, u_data, y_data):

    # Set parameter bounds based on data characteristics
    y_mean = np.mean(y_data)
    y_range = np.max(y_data) - np.min(y_data)
    u_mean = np.mean(u_data)
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


# Unit tests
# Load input-output data
data_dir = 'data'
filename = 'io_data_fopdt.csv'
input_output_data = pd.read_csv(os.path.join(data_dir, filename))
assert input_output_data.shape == (881, 4)
assert input_output_data.columns.tolist() == [
    'Output', 'Time', 'Input1', 'Input2'
]

# Prepare input-output data
input_col = 'Input1'
t = input_output_data['Time'].to_numpy()
u_data = input_output_data[input_col].to_numpy()
y_data = input_output_data['Output'].to_numpy()

# Determine average time interval
time_step_sizes = np.diff(t)
dt = np.mean(time_step_sizes)
assert np.max(np.abs(time_step_sizes - dt)) < dt / 10

# Make simulation function
simulate = make_simulate_function(dt, u_data)

# Test simulate function
K = -10.0
tau = 400.0
n_delay = 2  # must be integer
y_base = 500.0
u_base = 5.0
y_init = 500.0

y_model = simulate(K, tau, n_delay, y_base, u_base, y_init)
assert y_model.shape == u_data.shape
assert np.all(np.isnan(y_model[:n_delay+1]))
assert np.all(~np.isnan(y_model[n_delay+1:]))

err = rms_prediction_error(y_model[n_delay+1:], y_data[n_delay+1:])
assert np.isclose(err, 31.153522652859007)

bounds = calculate_reasonable_bounds(t, u_data, y_data)
var_names = list(bounds.keys())
bounds = list(bounds.values())
