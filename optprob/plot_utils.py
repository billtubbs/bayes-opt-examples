import numpy as np
import matplotlib.pyplot as plt


def convergence_plot(problem, title=None):
    if title is None:
        title = f'Optimizer Convergence - {problem.name}'
    fevals = np.array([item[0] for item in problem.guesses])
    fig = plt.figure(figsize=(7, 2.5))
    ax = fig.gca()
    plt.semilogy(fevals, marker='.')
    plt.xlabel('Number of function evaluations')
    plt.ylabel('f(x)')
    plt.grid()
    plt.title(title)
    return ax


def convergence_plot_n_repeats(
    fun_evals, title=None, marker='auto', color='tab:blue', alpha=0.25
):

    n_repeats = len(fun_evals)

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
    if marker == 'auto':
        marker = '.' if max_len < 20 else None
    x = np.arange(max_len)
    ax.semilogy(x, median_vals, marker=marker, color=color, label='Median')

    # Fill between min and max
    ax.fill_between(
        x, min_vals, max_vals, alpha=alpha, color=color, label='Min-Max Range'
    )

    plt.xlabel('Number of function evaluations')
    plt.ylabel('f(x)')
    plt.grid()
    plt.title(title)
    plt.legend()
    
    return ax
