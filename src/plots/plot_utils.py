from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt

from src.core.utils import (
    CIStats,
    PoolingType,
)
from src.plots.plot_settings import (
    FIGURE_FORMATS,
    TEXT_WIDTH,
)


class ErrorBars(NamedTuple):
    ci_lowers: list[float]
    ci_uppers: list[float]


def format_name(
    name: str,
) -> str:
    return name.replace('_', ' ').title()


def format_pooling_type(
    pooling_type: PoolingType,
) -> str:
    return f'{format_name(pooling_type)} pooling'


def get_ax(
    ax: plt.Axes | None,
    figsize: tuple[float, float],
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def get_offset(
    idx: int,
    num: int,
    width: float,
) -> float:
    offset = (idx - (num - 1) / 2) * width
    return offset


def get_error_bars(
    ci_stats_list: list[CIStats],
) -> ErrorBars:
    ci_lowers = [
        max(ci_stats['estimate'] - ci_stats['ci_lower'], 0.0)
        for ci_stats in ci_stats_list
    ]
    ci_uppers = [
        max(ci_stats['ci_upper'] - ci_stats['estimate'], 0.0)
        for ci_stats in ci_stats_list
    ]
    error_bars = ErrorBars(ci_lowers, ci_uppers)
    return error_bars


def plot_ci_points(
    ax: plt.Axes,
    positions: list[float],
    ci_stats_list: list[CIStats],
    horizontal: bool = False,
    **errorbar_kwargs,
) -> None:
    estimates = [ci_stats['estimate'] for ci_stats in ci_stats_list]
    error_bars = get_error_bars(ci_stats_list)
    if horizontal:
        ax.errorbar(estimates, positions, xerr=error_bars, **errorbar_kwargs)
    else:
        ax.errorbar(positions, estimates, yerr=error_bars, **errorbar_kwargs)


def set_row_labels(
    ax: plt.Axes,
    labels: list[str],
) -> None:
    ax.set_yticks(range(len(labels)), labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def save_figure(
    fig: plt.Figure,
    filename: str,
    out_dir: Path,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        fig.savefig(out_dir / f'{filename}.{figure_format}', bbox_inches='tight')
    plt.close(fig)


def plot_and_save(
    plot_fn: Callable[..., plt.Axes],
    panels: list[tuple],
    filename: str,
    out_dir: Path,
    width: float = TEXT_WIDTH,
    height_ratio: float = 0.35,
    sharey: bool = False,
    legend_on_last: bool = False,
    **plot_kwargs,
) -> None:
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, height_ratio * width), sharey=sharey, squeeze=False,
    )
    for ax, panel_args in zip(axes.flat, panels):
        plot_fn(*panel_args, ax=ax, **plot_kwargs)
    if sharey:
        for ax in axes.flat[1:]:
            ax.set_ylabel('')
    if legend_on_last:
        handles_by_label = {}
        for ax in axes.flat:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles_by_label.setdefault(label, handle)
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label),
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles_by_label),
            frameon=False,
        )
    save_figure(fig, filename, out_dir)
