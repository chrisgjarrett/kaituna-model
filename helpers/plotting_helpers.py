import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def visualise_results(y_actual, y_fit, y_pred):

    """Visualising training/test results"""

    # Plotting helpers
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
    )

    palette = dict(palette='husl', n_colors=64)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    ax1 = y_actual[y_fit.index].plot(ax=ax1)
    ax1 = _plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
    _ = ax1.legend(['Flowrate (train)', 'Forecast'])

    ax2 = y_actual[y_pred.index].plot(**plot_params, ax=ax2)
    ax2 = _plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
    _ = ax2.legend(['Flowrate (test)', 'Forecast'])

    plt.xlabel("Date")
    plt.ylabel('Flowrate (cumecs)')
    plt.legend(["Actual","Training","Validation"])
    plt.show()


def _plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    """Plots a multistep prediction"""

    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax
