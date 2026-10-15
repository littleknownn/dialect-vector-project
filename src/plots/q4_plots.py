import matplotlib.pyplot as plt

from src.core.utils import (
    Q4Results,
    Q4Stratum,
)
from src.plots.plot_utils import (
    DEFAULT_FIGSIZE,
    format_name,
    get_error_bars,
)


def plot_mean_projection_by_stratum(
    q4_results: Q4Results,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    strata_list = list(Q4Stratum)
    stratum_stats = [
        q4_results['mean_projection_by_stratum'][stratum]
        for stratum in strata_list
    ]
    x_ticks = range(len(strata_list))

    ax.bar(
        x_ticks,
        [ci_stats['estimate'] for ci_stats in stratum_stats],
        yerr=get_error_bars(stratum_stats),
        color='blue',
    )

    x_tick_labels = [
        f'{format_name(stratum)}\n(n={q4_results['num_pairs_by_stratum'][stratum]:,})'.replace(' ', '\n')
        for stratum in Q4Stratum
    ]
    ax.set_title(f'{format_name(q4_results['pooling_type'])} Pooling')
    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_ylabel('Mean projection onto held-out vector')
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_group_shares(
    q4_results: Q4Results,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    group_names = list(q4_results['group_shares'])
    group_stats = [q4_results['group_shares'][group_name] for group_name in group_names]
    colors = [
        'green' if group_name in q4_results['contributing_groups'] else 'gray'
        for group_name in group_names
    ]
    y_ticks = range(len(group_names))

    ax.barh(
        y_ticks,
        [ci_stats['estimate'] for ci_stats in group_stats],
        xerr=get_error_bars(group_stats),
        color=colors,
    )
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    ax.set_title(f'{format_name(q4_results['pooling_type'])} Pooling, R Squared = {q4_results['r_squared']:.2f}')
    ax.set_xlabel('Share of mean projection')
    ax.set_yticks(y_ticks, [format_name(group_name) for group_name in group_names])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    return ax