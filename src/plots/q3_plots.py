import matplotlib.pyplot as plt

from src.core.utils import (
    PoolingType,
    Q3EditCount,
    Q3Results,
)
from src.plots.plot_settings import (
    BELOW_AXES_LEGEND_KWARGS,
    DEFAULT_FIGSIZE,
    EDIT_COUNT_LABELS,
    FOREST_WIDTH,
    MEAN_PROJECTION_LABEL,
    PARTIAL_SPEARMAN_LABEL,
    POOLING_COLORS,
    Q3_MIN_COEF,
    REFERENCE_LINE_KWARGS,
    THRESHOLD_LINE_KWARGS,
    ZERO_REFERENCE,
)
from src.plots.plot_utils import (
    format_pooling_type,
    get_ax,
    get_offset,
    plot_ci_points,
    set_row_labels,
)


def plot_q3_by_bin(
    q3_results: Q3Results,
    edit_count: Q3EditCount = Q3EditCount.TOTAL,
    ax: plt.Axes | None = None,
    show_title: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    edit_count_labels: dict[Q3EditCount, str] = EDIT_COUNT_LABELS,
    mean_projection_label: str = MEAN_PROJECTION_LABEL,
) -> plt.Axes:
    ax = get_ax(ax, figsize)

    count_results = q3_results['count_results'][edit_count]
    ci_stats_by_bin = count_results['projection_mean_ci_by_bin']
    bin_ids = sorted(ci_stats_by_bin, key=int)
    bin_numbers = [int(bin_id) + 1 for bin_id in bin_ids]
    partial_spearman = count_results['partial_spearman_ci']
    plot_ci_points(
        ax,
        bin_numbers,
        [ci_stats_by_bin[bin_id] for bin_id in bin_ids],
        marker='o',
        color=pooling_colors[q3_results['pooling_type']],
        label=(
            f'Partial $\\rho$ = {partial_spearman['estimate']:.2f} '
            f'[{partial_spearman['ci_lower']:.2f}, {partial_spearman['ci_upper']:.2f}]'
        ),
    )

    if show_title:
        ax.set_title(format_pooling_type(q3_results['pooling_type']))
    ax.set_xlabel(f'{edit_count_labels[edit_count]} per word (bin)')
    ax.set_xticks(bin_numbers)
    ax.set_ylabel(mean_projection_label)
    ax.legend(loc='lower right')
    return ax


def plot_q3_levels(
    q3_results_list: list[Q3Results],
    min_coef: float = Q3_MIN_COEF,
    edit_counts: tuple[Q3EditCount, ...] = tuple(Q3EditCount),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    edit_count_labels: dict[Q3EditCount, str] = EDIT_COUNT_LABELS,
    zero_reference: float = ZERO_REFERENCE,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    threshold_line_kwargs: dict[str, str | float] = THRESHOLD_LINE_KWARGS,
    partial_spearman_label: str = PARTIAL_SPEARMAN_LABEL,
    below_axis_legend_kwargs: dict[str, str | int | tuple[float, float]] = BELOW_AXES_LEGEND_KWARGS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    for pooling_idx, results in enumerate(q3_results_list):
        offset = get_offset(pooling_idx, len(q3_results_list), forest_width)
        plot_ci_points(
            ax,
            [row_idx + offset for row_idx in range(len(edit_counts))],
            [results['count_results'][edit_count]['partial_spearman_ci'] for edit_count in edit_counts],
            horizontal=True,
            fmt='o',
            color=pooling_colors[results['pooling_type']],
            label=format_pooling_type(results['pooling_type']),
        )
    ax.axvline(zero_reference, **reference_line_kwargs)
    ax.axvline(min_coef, **threshold_line_kwargs)
    set_row_labels(ax, [edit_count_labels[edit_count] for edit_count in edit_counts])
    ax.set_xlabel(partial_spearman_label)
    ax.legend(**below_axis_legend_kwargs)
    return ax
