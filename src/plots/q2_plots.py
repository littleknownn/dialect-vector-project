import matplotlib.pyplot as plt

from src.core.utils import (
    AaeFeature,
    DifferenceVector,
    PoolingType,
    Q2Family,
    Q2Results,
)
from src.plots.data_loaders import select_q2_results
from src.plots.plot_settings import (
    AUROC_LABEL,
    BASELINE_COLORS,
    BASELINE_LABELS,
    BELOW_AXES_LEGEND_KWARGS,
    CHANCE_AUROC,
    CORPUS_LABELS,
    DEFAULT_FIGSIZE,
    FOREST_WIDTH,
    MDA_MARKER,
    NULL_BAND_KWARGS,
    POOLING_COLORS,
    PROBE_SET_LABELS,
    REFERENCE_LINE_KWARGS,
    THRESHOLD_LINE_KWARGS,
)
from src.plots.plot_utils import (
    format_pooling_type,
    get_ax,
    get_offset,
    plot_ci_points,
    set_row_labels,
)


def _plot_auroc_row(
    ax: plt.Axes,
    row_position: float,
    results: Q2Results,
    color: str,
    label: str | None,
    forest_width: float = FOREST_WIDTH,
    null_band_kwargs: dict[str, str | float] = NULL_BAND_KWARGS,
    mda_marker: str = MDA_MARKER,
) -> None:
    plot_ci_points(
        ax, [row_position], [results['auroc_ci']], horizontal=True, fmt='o', color=color, label=label,
    )
    if results['null_band'] is not None:
        ax.fill_betweenx(
            [
                row_position - forest_width / 2,
                row_position + forest_width / 2,
            ],
            *results['null_band'],
            **null_band_kwargs,
        )


def _format_auroc_axis(
    ax: plt.Axes,
    row_labels: list[str],
    min_auroc: float,
    chance_auroc: float = CHANCE_AUROC,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    threshold_line_kwargs: dict[str, str | float] = THRESHOLD_LINE_KWARGS,
    auroc_label: str = AUROC_LABEL,
    below_axes_legend_kwargs: dict = BELOW_AXES_LEGEND_KWARGS,
) -> None:
    ax.axvline(chance_auroc, **reference_line_kwargs)
    ax.axvline(min_auroc, **threshold_line_kwargs)
    set_row_labels(ax, row_labels)
    ax.set_xlabel(auroc_label)
    ax.legend(**below_axes_legend_kwargs)


def plot_q2_primary(
    q2_results_list: list[Q2Results],
    corpus_names: tuple[str, ...],
    pooling_types: tuple[PoolingType, ...],
    min_auroc: float,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    corpus_labels: dict[str, str] = CORPUS_LABELS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    for pooling_idx, pooling_type in enumerate(pooling_types):
        offset = get_offset(pooling_idx, len(pooling_types), forest_width)
        for row_idx, corpus_name in enumerate(corpus_names):
            results, = select_q2_results(
                q2_results_list, corpus_name=corpus_name, family=Q2Family.PRIMARY, pooling_type=pooling_type,
            )
            label = (
                format_pooling_type(pooling_type)
                if row_idx == 0
                else None
            )
            _plot_auroc_row(ax, row_idx + offset, results, pooling_colors[pooling_type], label)
    _format_auroc_axis(
        ax, [corpus_labels[corpus_name] for corpus_name in corpus_names], min_auroc,
    )
    return ax


def plot_q2_by_definition(
    q2_results_list: list[Q2Results],
    corpus_name: str,
    layer: int,
    pooling_types: tuple[PoolingType, ...],
    min_auroc: float,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    probe_set_labels: dict[tuple[AaeFeature, ...], str] = PROBE_SET_LABELS,
    corpus_labels: dict[str, str] = CORPUS_LABELS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    corpus_results = select_q2_results(
        q2_results_list, corpus_name=corpus_name, layer=layer, vector=DifferenceVector.DIALECT,
    )
    rows = [(list(probes), False) for probes in probe_set_labels] + [(list(AaeFeature), True)]
    row_labels = list(probe_set_labels.values()) + ['Any feature, no filter-flagged words']
    for pooling_idx, pooling_type in enumerate(pooling_types):
        offset = get_offset(pooling_idx, len(pooling_types), forest_width)
        for row_idx, (probes, excludes_filter_flagged_words) in enumerate(rows):
            results, = select_q2_results(
                corpus_results,
                pooling_type=pooling_type,
                positive_probes=list(probes),
                excludes_filter_flagged_words=excludes_filter_flagged_words,
            )
            label = (
                format_pooling_type(pooling_type)
                if row_idx == 0
                else None
            )
            _plot_auroc_row(ax, row_idx + offset, results, pooling_colors[pooling_type], label)

    midpoint_results = select_q2_results(corpus_results, pooling_type=PoolingType.BOUNDARY)
    if midpoint_results:
        row_labels.append('Any feature, midpoint read')
        _plot_auroc_row(
            ax,
            len(row_labels) - 1,
            midpoint_results[0],
            pooling_colors[PoolingType.BOUNDARY],
            'Midpoint read',
        )
    _format_auroc_axis(ax, row_labels, min_auroc)
    ax.set_title(corpus_labels[corpus_name])
    return ax


def plot_q2_layer_sweep(
    q2_results_list: list[Q2Results],
    corpus_name: str,
    pooling_types: tuple[PoolingType, ...],
    min_auroc: float,
    primary_layer: int | None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    chance_auroc: float = CHANCE_AUROC,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    threshold_line_kwargs: dict[str, str | float] = THRESHOLD_LINE_KWARGS,
    auroc_label: str = AUROC_LABEL,
    corpus_labels: dict[str, str] = CORPUS_LABELS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    for pooling_type in pooling_types:
        pooling_results = sorted(
            select_q2_results(
                q2_results_list,
                corpus_name=corpus_name,
                pooling_type=pooling_type,
                vector=DifferenceVector.DIALECT,
                positive_probes=list(AaeFeature),
                excludes_filter_flagged_words=False,
            ),
            key=lambda results: results['layer'],
        )
        plot_ci_points(
            ax,
            [results['layer'] for results in pooling_results],
            [results['auroc_ci'] for results in pooling_results],
            marker='o',
            color=pooling_colors[pooling_type],
            label=format_pooling_type(pooling_type),
        )

    ax.axhline(chance_auroc, **reference_line_kwargs)
    ax.axhline(min_auroc, **threshold_line_kwargs)
    if primary_layer is not None:
        ax.axvline(primary_layer, **threshold_line_kwargs)

    ax.set_title(corpus_labels[corpus_name])
    ax.set_xlabel('Layer')
    ax.set_ylabel(auroc_label)

    ax.legend()
    return ax


def plot_q2_baselines(
    q2_results_list: list[Q2Results],
    corpus_names: tuple[str, ...],
    pooling_type: PoolingType,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    baseline_labels: dict[str, str] = BASELINE_LABELS,
    baseline_colors: dict[str, str] = BASELINE_COLORS,
    chance_auroc: float = CHANCE_AUROC,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    auroc_label: str = AUROC_LABEL,
    corpus_labels: dict[str, str] = CORPUS_LABELS,
    bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    for row_idx, corpus_name in enumerate(corpus_names):
        results, = select_q2_results(
            q2_results_list,
            corpus_name=corpus_name,
            family=Q2Family.PRIMARY,
            pooling_type=pooling_type,
        )
        ax.plot(
            results['auroc_ci']['estimate'],
            row_idx,
            marker='o',
            color=pooling_colors[pooling_type],
            label=(
                'Difference direction'
                if row_idx == 0
                else None
            ),
        )
        for feature_idx, (feature_name, feature_label) in enumerate(baseline_labels.items()):
            offset = get_offset(feature_idx, len(baseline_labels), forest_width)
            ax.plot(
                results['baseline_aurocs'][feature_name],
                row_idx + offset,
                marker='x',
                linestyle='',
                color=baseline_colors[feature_name],
                label=(
                    feature_label
                    if row_idx == 0
                    else None
                ),
            )
    ax.axvline(chance_auroc, **reference_line_kwargs)
    set_row_labels(ax, [corpus_labels[corpus_name] for corpus_name in corpus_names])
    ax.set_xlabel(auroc_label)
    ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
    return ax
