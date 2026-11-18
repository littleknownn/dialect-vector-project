from pathlib import Path

from src.core.utils import (
    CIStats,
    Q1Results,
    Q2Family,
    Q2Results,
    Q3Results,
    Q4CurveFunction,
    Q4EditKind,
    Q4Results,
)
from src.plots.plot_settings import (
    CORPUS_LABELS,
    DIFFERENCE_VECTOR_LABELS,
    EDIT_COUNT_LABELS,
    EDIT_KIND_LABELS,
    PROBE_SET_LABELS,
)
from src.plots.plot_utils import format_name


MISSING_VALUE = '--'


def format_ci(
    ci_stats: CIStats | None,
    num_digits: int = 2,
) -> str:
    if ci_stats is None:
        return MISSING_VALUE
    return f'{ci_stats['estimate']:.{num_digits}f} [{ci_stats['ci_lower']:.{num_digits}f}, {ci_stats['ci_upper']:.{num_digits}f}]'


def format_p_value(
    p_value: float | None,
) -> str:
    if p_value is None:
        return MISSING_VALUE
    if p_value < 0.001:
        return '$<$.001'
    return f'{p_value:.3f}'.lstrip('0')


def write_table(
    name: str,
    header: list[str],
    rows: list[list[str]],
    caption: str,
    out_dir: Path,
) -> None:
    lines = [ # LLM-Generated LaTeX Table Setup
        r'\begin{table*}[p]', r'\centering', r'\small',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l' + 'r' * (len(header) - 1) + '}', r'\toprule',
        ' & '.join(header) + r' \\', r'\midrule',
        *(' & '.join(row) + r' \\' for row in rows),
        r'\bottomrule', r'\end{tabular}}',
        rf'\caption{{{caption}}}', rf'\label{{tab:{name}}}', r'\end{table*}',
    ]
    table_dir = out_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def write_q1_table(
    q1_results_list: list[Q1Results],
    layer: int,
    out_dir: Path,
) -> None:
    rows = [
        [
            DIFFERENCE_VECTOR_LABELS[results['vector']],
            format_name(results['pooling_type']),
            f'{results['num_pairs']:,}',
            format_ci(results['positive_projection_ratio_ci']),
            format_ci(results['standardized_projection_mean_ci']),
            format_p_value(results['sign_flip_p_value_fdr']),
            format_name(results['label']),
        ]
        for results in q1_results_list
        if results['layer'] == layer
    ]
    header = ['Vector', 'Pooling', 'Pairs', 'Positive share [95\\% CI]', 'Standardized mean [95\\% CI]', 'BH $p$', 'Label']
    write_table('q1', header, rows, 'Testing pair consistency on $A$.', out_dir)


def write_q2_table(
    q2_results_list: list[Q2Results],
    out_dir: Path,
) -> None:
    rows = [
        [
            CORPUS_LABELS[results['corpus_name']],
            PROBE_SET_LABELS[tuple(results['positive_probes'])] + (', no filter-flagged' if results['excludes_filter_flagged_words'] else ''),
            format_name(results['family']),
            format_name(results['pooling_type']),
            f'{results['num_positives']:,}',
            format_ci(results['auroc_ci']),
            format_p_value(results['null_p_value_fdr']),
            f'{results['min_detectable_auroc']:.2f}',
            format_name(results['label']),
        ]
        for family in (Q2Family.PRIMARY, Q2Family.SECONDARY)
        for results in q2_results_list
        if results['family'] == family
    ]
    header = ['Corpus', 'Feature detectors', 'Family', 'Pooling', 'Positives', 'AUROC [95\\% CI]', 'BH $p$', 'MDA', 'Label']
    caption = 'MDA: minimum detectable AUROC at 80\\% power.'
    write_table('q2', header, rows, caption, out_dir)


def write_q3_table(
    q3_results_list: list[Q3Results],
    out_dir: Path,
) -> None:
    rows = [
        [
            EDIT_COUNT_LABELS[edit_count],
            format_name(results['pooling_type']),
            format_ci(count_results['partial_spearman_ci']),
            format_p_value(count_results['partial_spearman_p_value_fdr']),
        ]
        for results in q3_results_list
        for edit_count, count_results in results['count_results'].items()
    ]
    header = ['Level', 'Pooling', 'Partial $\\rho$ [95\\% CI]', 'BH $p$']
    caption = 'Edit correlation with projection onto $A$ by edit level.'
    write_table('q3', header, rows, caption, out_dir)


def write_q4_table(
    q4_results: Q4Results,
    out_dir: Path,
    curves: tuple[Q4CurveFunction, ...] = (Q4CurveFunction.LOG, Q4CurveFunction.CEILING),
) -> None:
    verdicts = q4_results['verdicts'] or {}
    rows = [
        [
            EDIT_KIND_LABELS[edit_kind],
            f'{q4_results['num_edits_by_kind'].get(edit_kind, 0):,}',
            *(format_ci(q4_results['curve_results'][curve]['weights'].get(edit_kind)) for curve in curves),
            format_name(verdicts.get(edit_kind, MISSING_VALUE)),
        ]
        for edit_kind in Q4EditKind
    ]
    pooling_type = q4_results['pooling_type']
    header = ['Edit kind', 'Edits', 'Weight, log', 'Weight, ceiling', 'Verdict']
    caption = f'Weight by edit kind ({pooling_type} pooling).'
    write_table(f'q4_{pooling_type}', header, rows, caption, out_dir)


def write_replication_table(
    model_rows: list[tuple[str, str, int, list[Q1Results], list[Q2Results]]],
    out_dir: Path,
    coraal_name: str = 'coraal_merged',
) -> None:
    def _consistency(q1_results_list, layer, pooling_type):
        results = next(
            r for r in q1_results_list
            if r['vector'] == ''
            and r['layer'] == layer
            and r['pooling_type'] == pooling_type
        )
        return (
            f"{results['nonidentical_pairs_positive_projection_ratio_ci']['estimate']:.3f} / "
            f"{results['nonidentical_pairs_standardized_projection_mean_ci']['estimate']:.2f}"
        )

    def _transfer(q2_results_list, pooling_type):
        results = next(
            r for r in q2_results_list
            if r['corpus_name'] == coraal_name
            and r['family'] == Q2Family.PRIMARY
            and r['pooling_type'] == pooling_type
        )
        return format_ci(results['auroc_ci'])

    body = [
        f'{model} & {layer_label} & '
        f"{_consistency(q1, layer, 'mean')} & {_consistency(q1, layer, 'last')} & "
        f"{_transfer(q2, 'mean')} & {_transfer(q2, 'last')}" + r' \\'
        for model, layer_label, layer, q1, q2 in model_rows
    ]
    lines = [
        r'\begin{table*}[t]', r'\centering', r'\small',
        r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
        r' & & \multicolumn{2}{c}{Consistency of $A$: positive share / standardized mean} & \multicolumn{2}{c}{CORAAL AUROC [95\% CI]} \\',
        r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
        r'Model & Layer & Mean & Last & Mean & Last \\', r'\midrule',
        *body,
        r'\bottomrule', r'\end{tabular}',
        r"\caption{Replication across models (testing pairs; each model's own $A$).}",
        r'\label{tab:replication}', r'\end{table*}',
    ]
    table_dir = out_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / 'replication.tex').write_text('\n'.join(lines) + '\n')
