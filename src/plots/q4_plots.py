import matplotlib.pyplot as plt

from src.core.utils import (
    PoolingType,
    Q4CurveFunction,
    Q4EditKind,
    Q4EditVerdict,
    Q4Results,
    Q4RobustnessResults,
    Q4RobustnessVariant,
    Q4Stratum,
)
from src.plots.plot_settings import (
    CURVE_LABELS,
    CURVE_MARKERS,
    DEFAULT_FIGSIZE,
    EDIT_KIND_LABELS,
    FOREST_WIDTH,
    MEAN_PROJECTION_LABEL,
    POOLING_COLORS,
    REFERENCE_LINE_KWARGS,
    ROBUSTNESS_VARIANT_LABELS,
    SECONDARY_ALPHA,
    STRATUM_LABELS,
    UNIT_WEIGHT,
    WEIGHT_LABEL,
    ZERO_REFERENCE,
)
from src.plots.plot_utils import (
    format_pooling_type,
    get_ax,
    get_error_bars,
    get_offset,
    plot_ci_points,
    set_row_labels,
)


def plot_q4_strata(
    q4_results: Q4Results,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    stratum_labels: dict[Q4Stratum, str] = STRATUM_LABELS,
    zero_reference: float = ZERO_REFERENCE,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    mean_projection_label: str = MEAN_PROJECTION_LABEL,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    strata_list = list(Q4Stratum)
    stratum_stats = [
        q4_results['mean_projection_by_stratum'][stratum]
        for stratum in strata_list
    ]

    ax.bar(
        range(len(strata_list)),
        [ci_stats['estimate'] for ci_stats in stratum_stats],
        yerr=get_error_bars(stratum_stats),
        color=pooling_colors[q4_results['pooling_type']],
    )
    ax.axhline(zero_reference, **reference_line_kwargs)
    
    ax.set_title(format_pooling_type(q4_results['pooling_type']))    
    tick_labels = [
        stratum_labels[stratum].replace(' ', '\n') + f'\n{q4_results['num_pairs_by_stratum'][stratum]:,}'
        for stratum in strata_list
    ]
    ax.set_xlabel('Stratum (pairs)')
    ax.set_xticks(range(len(strata_list)), tick_labels)
    ax.tick_params(axis='x', labelsize='small')
    ax.set_ylabel(mean_projection_label)
    return ax


def plot_q4_weights(
    q4_results: Q4Results,
    curves: tuple[Q4CurveFunction, ...] = (Q4CurveFunction.LOG, Q4CurveFunction.CEILING),
    unit_kind: Q4EditKind = Q4EditKind.CONTENT_CHANGE,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    forest_width: float = FOREST_WIDTH,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    curve_markers: dict[Q4CurveFunction, str] = CURVE_MARKERS,
    curve_labels: dict[Q4CurveFunction, str] = CURVE_LABELS,
    edit_kind_labels: dict[Q4EditKind, str] = EDIT_KIND_LABELS,
    unit_weight: float = UNIT_WEIGHT,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    weight_label: str = WEIGHT_LABEL,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    color = pooling_colors[q4_results['pooling_type']]
    verdicts = q4_results['verdicts'] or {}
    edit_kinds = [
        edit_kind
        for edit_kind in Q4EditKind
        if edit_kind != unit_kind
        and edit_kind in q4_results['curve_results'][curves[0]]['weights']
    ]

    for curve_idx, curve in enumerate(curves):
        offset = get_offset(curve_idx, len(curves), forest_width)
        weights = q4_results['curve_results'][curve]['weights']
        for row_idx, edit_kind in enumerate(edit_kinds):
            is_different = verdicts.get(edit_kind) in (Q4EditVerdict.STRONGER, Q4EditVerdict.WEAKER)
            plot_ci_points(
                ax,
                [row_idx + offset],
                [weights[edit_kind]],
                horizontal=True,
                fmt=curve_markers[curve],
                color=color,
                markerfacecolor=(
                    color
                    if is_different
                    else 'white'
                ),
                label=(
                    curve_labels[curve]
                    if row_idx == 0
                    else None
                ),
            )
    ax.axvline(unit_weight, **reference_line_kwargs)
    
    ax.set_title(format_pooling_type(q4_results['pooling_type']))
    ax.set_xlabel(weight_label)
    set_row_labels(ax, [edit_kind_labels[edit_kind] for edit_kind in edit_kinds])
    return ax


def plot_q4_shares(
    q4_results: Q4Results,
    curve: Q4CurveFunction = Q4CurveFunction.LOG,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    edit_kind_labels: dict[Q4EditKind, str] = EDIT_KIND_LABELS,
    curve_labels: dict[Q4CurveFunction, str] = CURVE_LABELS,
    secondary_alpha: float = SECONDARY_ALPHA,
    zero_reference: float = ZERO_REFERENCE,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    curve_results = q4_results['curve_results'][curve]
    color = pooling_colors[q4_results['pooling_type']]
    edit_kinds = [
        edit_kind
        for edit_kind in Q4EditKind
        if edit_kind in curve_results['shares']
    ]
    share_stats = [curve_results['shares'][edit_kind] for edit_kind in edit_kinds]
    other_shares = {
        'Format flags': curve_results['format_flags_share'],
        'Pair controls': curve_results['pair_controls_share'],
        'Unexplained': curve_results['unexplained_share'],
    }

    ax.barh(
        range(len(edit_kinds)),
        [ci_stats['estimate'] for ci_stats in share_stats],
        xerr=get_error_bars(share_stats),
        color=color,
    )
    ax.barh(
        range(len(edit_kinds), len(edit_kinds) + len(other_shares)),
        list(other_shares.values()),
        color=color,
        alpha=secondary_alpha,
    )
    ax.axvline(zero_reference, **reference_line_kwargs)

    ax.set_title(f'{format_pooling_type(q4_results['pooling_type'])}, {curve_labels[curve].lower()}')
    ax.set_xlabel('Share of mean projection')
    set_row_labels(
        ax,
        [edit_kind_labels[edit_kind] for edit_kind in edit_kinds] + list(other_shares),
    )
    return ax


def plot_q4_calibration(
    q4_results: Q4Results,
    curves: tuple[Q4CurveFunction, ...] = (Q4CurveFunction.LOG, Q4CurveFunction.CEILING),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    curve_markers: dict[Q4CurveFunction, str] = CURVE_MARKERS,
    curve_labels: dict[Q4CurveFunction, str] = CURVE_LABELS,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    color = pooling_colors[q4_results['pooling_type']]
    all_values = []

    for curve_idx, curve in enumerate(curves):
        curve_results = q4_results['curve_results'][curve]
        predicted, observed = zip(*curve_results['calibration'])
        all_values.extend(predicted + observed)
        ax.plot(
            predicted,
            observed,
            marker=curve_markers[curve],
            linestyle='',
            color=color,
            markerfacecolor=color if curve_idx == 0 else 'white',
            label=f'{curve_labels[curve]} (held-out $R^2$ = {curve_results['held_out_r_squared']:.2f})',
        )
    axis_limits = (min(all_values), max(all_values))
    ax.plot(axis_limits, axis_limits, **reference_line_kwargs)

    ax.set_title(format_pooling_type(q4_results['pooling_type']))
    ax.set_xlabel('Mean predicted $p_i$ (bin)')
    ax.set_ylabel('Mean observed $p_i$')
    ax.legend(loc='upper left')
    return ax


def plot_q4_robustness(
    robustness_results_list: list[Q4RobustnessResults],
    q4_results: Q4Results,
    curve: Q4CurveFunction = Q4CurveFunction.LOG,
    edit_kind: Q4EditKind = Q4EditKind.GRAMMAR_WORD,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    pooling_colors: dict[PoolingType, str] = POOLING_COLORS,
    robustness_variant_labels: dict[Q4RobustnessVariant, str] = ROBUSTNESS_VARIANT_LABELS,
    unit_weight: float = UNIT_WEIGHT,
    secondary_alpha: float = SECONDARY_ALPHA,
    reference_line_kwargs: dict[str, str | float] = REFERENCE_LINE_KWARGS,
    weight_label: str = WEIGHT_LABEL,
) -> plt.Axes:
    ax = get_ax(ax, figsize)
    curve_results_list = [
        results
        for results in robustness_results_list
        if results['curve'] == curve
        and edit_kind in results['weights']
    ]
    row_labels = [
        f'{robustness_variant_labels[results['variant']]} {results['removed_word'] or ''}'.strip()
        for results in curve_results_list
    ]

    plot_ci_points(
        ax,
        list(range(len(curve_results_list))),
        [results['weights'][edit_kind] for results in curve_results_list],
        horizontal=True,
        fmt='o',
        color=pooling_colors[q4_results['pooling_type']],
    )
    primary_weight = q4_results['curve_results'][curve]['weights'][edit_kind]
    ax.axvspan(primary_weight['ci_lower'], primary_weight['ci_upper'], color='gray', alpha=secondary_alpha / 2, linewidth=0)
    ax.axvline(primary_weight['estimate'], color='gray', linewidth=0.6)
    ax.axvline(unit_weight, **reference_line_kwargs)

    ax.set_xlabel(weight_label)
    set_row_labels(ax, row_labels)
    return ax
