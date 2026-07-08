from IPython.display import display
import matplotlib.pyplot as plt


def plot_figure(fig: plt.Figure) -> None:
    display(fig)
    plt.close(fig)
