"""Specialised plot functions for visualising Bayesian optimization
process when using the bayes_opt Python package.  Some of these
I wrote myself but posterior and plot_gp are adapted from
similar functions used in the example notebooks by Fernando 
Nogueira available at https://github.com/fmfn/BayesianOptimization.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Functions to plot the Gaussian process and utility function

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    steps = len(optimizer.space)

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    ax = axes[0]
    ax.plot(x, y, linewidth=3, label='Target')
    ax.plot(x_obs.flatten(), y_obs, 'd', label=u'Observations', color='r')
    ax.plot(x, mu, '--', color='k', label='Prediction')

    ax.fill(np.concatenate([x, x[::-1]]), 
            np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
            alpha=.6, fc='c', ec='None', label='95% confidence interval')

    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')
    ax.set_title(f'Gaussian Process and Utility Function After {steps} Steps')
    ax.grid()
    ax.legend(bbox_to_anchor=(1, 0.5))

    from bayes_opt import UtilityFunction
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)

    ax = axes[1]
    ax.plot(x, utility, label='Utility function', color='purple')
    ax.plot(
        x[np.argmax(utility)], np.max(utility), '*', markersize=15,
        label=u'Next best guess', markerfacecolor='gold',
        markeredgecolor='k', markeredgewidth=1
    )
    ax.set_ylim((0, np.max(utility) + 0.5))
    ax.set_ylabel('Utility')
    ax.set_xlabel('x')
    ax.grid()
    ax.legend(bbox_to_anchor=(1, 0.5))

    return fig


# 2-D plotting functions

def generate_plot_data(f, xlim, ylim, nx=100, ny=100):
    X = np.linspace(*xlim, nx)
    Y = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    return X, Y, Z


def show_3d_surface_plot(X, Y, Z, zlim=(0, 100), title=None):
    """Make 3D surface plot"""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.gist_rainbow,
                           linewidth=0, antialiased=False)

    # Customize the axes.
    ax.set_zlim(*zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return ax


def show_contour_plot(
    X, Y, Z, minima=None, fmt='%.0f', txt_offset=(5, -3), title=None
):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.contour(X, Y, Z)
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=1, fmt=fmt, fontsize=10)
    if minima:
        # Show all the local minima as points
        for i, m in enumerate(minima):
            ax.plot(*m, 'ok')
            ax.annotate(
                f'$m_{i}$', m, xytext=txt_offset, textcoords='offset points'
            )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    ax.grid()
    return ax


def show_contour_plot_with_pt(X, Y, Z, pt, minima=None, fmt='%.0f',
                              txt_offset=(5, -3), title=None):
    # Show a countour plot with an additional point to
    # indicate the current solution
    ax = show_contour_plot(
        X, Y, Z, minima=minima, fmt=fmt, txt_offset=txt_offset, title=title
    )
    ax.plot(*pt, 'xr')
    ax.annotate(
        'pt', pt, xytext=txt_offset, color='r', textcoords='offset points'
    )
    return ax
