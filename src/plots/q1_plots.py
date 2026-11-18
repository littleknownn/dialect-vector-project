import matplotlib.pyplot as plt
import torch

from src.core.config_utils import Q1LabeledThresholds
from src.core.utils import (
    DifferenceVector,
    PairEditData,
    PoolingType,
    Q1Results,
    Q1Statistic,
)
from src.plots.data_loaders import select_q1_results
from src.plots.plot_settings import (
    DEFAULT_FIGSIZE,
    DIFFERENCE_VECTOR_LABELS,
    FOREST_WIDTH,
    HEADROOM_SCALAR,
    HISTOGRAM_ALPHA,
    POOLING_COLORS,
    PROJECTION_LABEL,
    REFERENCE_LINE_KWARGS,
    SUMMARY_TEXT_POS,
    THRESHOLD_LINE_KWARGS,
)
from src.plots.plot_utils import (
    format_pooling_type,
    get_ax,
    get_offset,
    plot_ci_points,
    set_row_labels,
)

_Q1_STATISTIC_LABELS = {
    Q1Statistic.POSITIVE_PROJECTION_RATIO: 'Positive share',
    Q1Statistic.STANDARDIZED_PROJECTION_MEAN: 'Standardized mean',
}
_Q1_THRESHOLD_FIELDS = {
    Q1Statistic.POSITIVE_PROJECTION_RATIO: 'positive_projection_ratio',
    Q1Statistic.STANDARDIZED_PROJECTION_MEAN: 'standardized_mean_ci_lower',
}


def plot_q1_histogram(
    projections: torch.Tensor,
    q1_results: Q1Results,
    pair_edit_data: list[PairEditData] | None = None,
    ax: plt.Axes | None = None,
    num_bins: int = 60,
    show_title: bool = True,
    show_legend: bool = True,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    histogram_alpha: float = HISTOGRAM_ALPHA,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    summary_text_pos: tuple[float, float] = SUMMARY_TEXT_POS,
    headroom_scalar: float = HEADROOM_SCALAR,
    projection_label: str = PROJECTION_LABEL,
) -> plt.Axes:
    ax = get_ax(ax, figsize)

    valued_pairs_mask = ~projections.isnan()
    group_masks = {'All pairs': valued_pairs_mask}
    if pair_edit_data is not None:
        ending_changed_mask = torch.tensor([pair['ending_changed'] for pair in pair_edit_data])
        group_masks = {
            'Ending changed': ending_changed_mask & valued_pairs_mask,
            'Ending unchanged': ~ending_changed_mask & valued_pairs_mask,
        }
    bin_edges = torch.histogram(projections[valued_pairs_mask].float(), bins=num_bins).bin_edges.numpy()
    for label, mask in group_masks.items():
        ax.hist(
            projections[mask].numpy(),
            bins=bin_edges,
            label=label,
            alpha=histogram_alpha,
        )

    ax.axvline(0, **reference_line_kwargs)

    positive_ratio = q1_results[Q1Statistic.POSITIVE_PROJECTION_RATIO]
    standardized_mean = q1_results[Q1Statistic.STANDARDIZED_PROJECTION_MEAN]
    summary = (
        f'Positive share: {positive_ratio['estimate']:.1%}\n'
        f'Standardized mean:\n'
        f'{standardized_mean['estimate']:.2f} '
        f'[{standardized_mean['ci_lower']:.2f}, {standardized_mean['ci_upper']:.2f}]'
    )
    ax.text(
        *summary_text_pos,
        summary,
        transform=ax.transAxes,
        va='top',
        bbox={
            'facecolor': 'white',
            'edgecolor': 'none',
            'alpha': 0.85,
            'pad': 1,
        },
    ) #

    if show_title:
        ax.set_title(format_pooling_type(q1_results['pooling_type']))
    ax.set_xlabel(projection_label)
    ax.set_ylabel('Pairs')
    _, upper_ylim = ax.get_ylim()
    ax.set_ylim(top=upper_ylim * headroom_scalar)
    if show_legend and pair_edit_data is not None:
        ax.legend(loc='upper right')
    return ax


def plot_q1_forest(
    q1_results_list: list[Q1Results],
    statistic: Q1Statistic,
    labeled_bounds: Q1LabeledThresholds,
    layer: int,
    vectors: tuple[DifferenceVector, ...] = tuple(DifferenceVector),
    pooling_types: tuple[PoolingType, ...] = tuple(PoolingType),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    threshold_line_kwargs: dict[str, str | float] = THRESHOLD_LINE_KWARGS,
    difference_vector_labels: dict[DifferenceVector, str] = DIFFERENCE_VECTOR_LABELS,
    q1_statistic_labels: dict[Q1Statistic, str] = _Q1_STATISTIC_LABELS,
    q1_threshold_fields: dict[Q1Statistic, str] = _Q1_THRESHOLD_FIELDS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)

    for pooling_idx, pooling_type in enumerate(pooling_types):
        offset = get_offset(pooling_idx, len(pooling_types), forest_width)
        row_positions, ci_stats_list = [], []

        for row_idx, vector in enumerate(vectors):
            selected_results = select_q1_results(q1_results_list, vector, layer=layer, pooling_type=pooling_type)
            if not selected_results:
                continue
            row_positions.append(row_idx + offset)
            results, = selected_results
            ci_stats_list.append(results[statistic])
        plot_ci_points(
            ax,
            row_positions,
            ci_stats_list,
            horizontal=True,
            fmt='o',
            color=pooling_colors[pooling_type],
            label=format_pooling_type(pooling_type),
        )
    for bounds in labeled_bounds:
        ax.axvline(getattr(bounds, q1_threshold_fields[statistic]), **threshold_line_kwargs)

    set_row_labels(
        ax,
        [difference_vector_labels[vector] for vector in vectors],
    )
    ax.set_xlabel(q1_statistic_labels[statistic])
    return ax


def plot_q1_layer_sweep(
    q1_results_list: list[Q1Results],
    statistic: Q1Statistic,
    vector: DifferenceVector = DifferenceVector.DIALECT,
    pooling_types: tuple[PoolingType, ...] = tuple(PoolingType),
    primary_layer: int | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    threshold_line_kwargs: dict[str, str | float] = THRESHOLD_LINE_KWARGS,
    q1_statistic_labels: dict[Q1Statistic, str] = _Q1_STATISTIC_LABELS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)

    for pooling_type in pooling_types:
        pooling_results = sorted(
            select_q1_results(q1_results_list, vector, pooling_type=pooling_type),
            key=lambda results: results['layer'],
        )
        plot_ci_points(
            ax,
            [results['layer'] for results in pooling_results],
            [results[statistic] for results in pooling_results],
            marker='o',
            color=pooling_colors[pooling_type],
            label=format_pooling_type(pooling_type),
        )
    if primary_layer is not None:
        ax.axvline(primary_layer, **threshold_line_kwargs)

    ax.set_xlabel('Layer')
    ax.set_ylabel(q1_statistic_labels[statistic])
    return ax
