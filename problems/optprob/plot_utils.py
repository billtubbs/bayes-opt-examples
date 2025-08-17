from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt


def function_evaluations_plot(
    problem, title=None, marker='.', markersize=3, linestyle='', **kwargs
):
    if title is None:
        title = f'Function Evaluations - {problem.name}'
    f_evals = np.array([f for f, x in problem.guesses])
    fig = plt.figure(figsize=(7, 2.5))
    ax = fig.gca()
    plt.semilogy(
        f_evals,
        marker=marker,
        markersize=markersize,
        linestyle=linestyle,
        **kwargs
    )
    plt.xlabel('Number of function evaluations')
    plt.ylabel('f(x)')
    plt.grid()
    plt.title(title)
    return ax


def best_guesses_plot(
    problem, title=None, marker='', linestyle='-', **kwargs
):
    if title is None:
        title = f'Best Guesses - {problem.name}'
    best_guesses = np.fromiter(
        accumulate([f for f, x in problem.guesses], min),
        dtype='float'
    )
    fig = plt.figure(figsize=(7, 2.5))
    ax = fig.gca()
    plt.semilogy(
        best_guesses,
        marker=marker,
        linestyle=linestyle,
        drawstyle='steps-post',
        **kwargs
    )
    plt.xlabel('Number of function evaluations')
    plt.ylabel('f(x) best guess')
    plt.grid()
    plt.title(title)
    return ax


def best_guesses_plot_n_repeats(
    fun_evals,
    color='tab:blue',
    alpha=0.25,
    best_guesses=True,
    title=None
):
    n_repeats = len(fun_evals)

    if best_guesses:
        fun_evals = [
            np.fromiter(accumulate(f, min), dtype='float')
            for f in fun_evals
        ]

    if title is None:
        title = f'Optimizer Convergence - {n_repeats} Iterations'

    # Find the maximum length among all sequences
    max_len = max(len(f) for f in fun_evals)

    # Extend all sequences to the same length by repeating their final value
    series_array = np.full((n_repeats, max_len), np.nan)
    for i, f in enumerate(fun_evals):
        n = len(f)
        series_array[i, :n] = f
        series_array[i, n:] = f[-1]

    # Calculate min, max, and median across all series at each iteration
    min_vals = np.min(series_array, axis=0)
    max_vals = np.max(series_array, axis=0)
    median_vals = np.median(series_array, axis=0)

    # Create the plot
    fig = plt.figure(figsize=(7, 2.5))
    ax = fig.gca()

    # Plot the median line
    x = np.arange(max_len)
    ax.semilogy(
        x,
        median_vals,
        marker='',
        linestyle='-',
        color=color,
        drawstyle='steps-post',
        label='Median',
    )

    # Fill between min and max
    ax.fill_between(
        x,
        min_vals,
        max_vals,
        alpha=alpha,
        color=color,
        label='Min-Max Range'
    )

    plt.xlabel('Number of function evaluations')
    plt.ylabel('f(x)')
    plt.grid()
    plt.title(title)
    plt.legend()
    
    return ax
