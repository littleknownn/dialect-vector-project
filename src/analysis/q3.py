from pathlib import Path
from typing import NamedTuple

from scipy.stats import (
    false_discovery_control,
    spearmanr,
)
import torch
from tqdm import tqdm

from src.analysis.pair_data import get_pooling_annotations_list
from src.core.config_utils import (
    ExperimentConfig,
    SupervisedPair,
)
from src.core.paths import Q3_RESULTS_PREFIX
import src.core.stats_utils as stats_utils
from src.core.utils import (
    PairAnnotations,
    PoolingType,
    Q3CountResults,
    Q3EditCount,
    Q3Results,
    Q4EditKind,
    SeedOffset,
    SplitSuffixes,
    TextPairAnnotations,
    save_json,
)


class DoseBins(NamedTuple):
    bin_ids: torch.Tensor
    bin_edges: list[float]
    has_zero_bin: bool


def _get_edit_count_bins(
    edit_counts: torch.Tensor,
    num_bins: int,
) -> DoseBins:
    is_zero = edit_counts == 0
    if is_zero.double().mean() < 1 / num_bins:
        bin_edges = torch.quantile(
            edit_counts,
            torch.linspace(0, 1, num_bins + 1, dtype=edit_counts.dtype)[1:-1],
        )
        bin_ids = torch.searchsorted(bin_edges, edit_counts)
        bins = DoseBins(bin_ids, bin_edges.tolist(), has_zero_bin=False)
        return bins

    nonzero_bin_edges = torch.quantile(
        edit_counts[~is_zero],
        torch.linspace(0, 1, num_bins, dtype=edit_counts.dtype)[1:-1],
    )
    bin_ids = torch.where(
        is_zero, 0, torch.searchsorted(nonzero_bin_edges, edit_counts) + 1,
    )
    bins = DoseBins(bin_ids, nonzero_bin_edges.tolist(), has_zero_bin=True)
    return bins


def _compute_edit_counts_dict(
    pooling_annotations_list: list[TextPairAnnotations],
    num_words: torch.Tensor,
    edits_by_count: dict[Q3EditCount, tuple[Q4EditKind, ...]],
) -> dict[Q3EditCount, torch.Tensor]:
    edit_counts_dict = {
        edit_count: torch.tensor(
            [
                sum(
                    edit['kind'] in edit_kinds
                    for edit in annotations['edits']
                )
                for annotations in pooling_annotations_list
            ],
            dtype=torch.float64,
        ) / num_words
        for edit_count, edit_kinds in edits_by_count.items()
    }
    edit_counts_dict[Q3EditCount.PROBE_COUNT] = torch.tensor(
        [
            annotations['num_aae_features']
            for annotations in pooling_annotations_list
        ],
        dtype=torch.float64,
    )
    return edit_counts_dict


def _compute_q3_count_results(
    projections: torch.Tensor,
    edit_counts: torch.Tensor,
    log_num_words: torch.Tensor,
    experiment_config: ExperimentConfig,
) -> Q3CountResults:
    bootstrap_fn = lambda values, statistic=None: stats_utils.compute_bootstrap_ci(
        values,
        statistic,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + SeedOffset.Q3_Q4_BOOTSTRAP,
    )
    values = torch.stack([projections, edit_counts, log_num_words], dim=1)
    bins = _get_edit_count_bins(edit_counts, experiment_config.q3.num_dose_bins)

    spearman_ci = bootstrap_fn(values, stats_utils.compute_spearman)
    spearman_p_value = float(spearmanr(projections, edit_counts).pvalue)
    partial_spearman_ci = bootstrap_fn(values, stats_utils.compute_partial_spearman)
    partial_spearman_p_value = stats_utils.compute_partial_spearman_p_value(
        float(stats_utils.compute_partial_spearman(values)), len(projections), num_covariates=1,
    )
    projection_mean_ci_by_bin = {
        bin_id: bootstrap_fn(projections[bins.bin_ids == bin_id])
        for bin_id in bins.bin_ids.unique().tolist()
    }
    count_results = Q3CountResults(
        spearman_ci=spearman_ci,
        spearman_p_value=spearman_p_value,
        spearman_p_value_fdr=None,
        partial_spearman_ci=partial_spearman_ci,
        partial_spearman_p_value=partial_spearman_p_value,
        partial_spearman_p_value_fdr=None,
        projection_mean_ci_by_bin=projection_mean_ci_by_bin,
        bin_edges=bins.bin_edges,
        has_zero_bin=bins.has_zero_bin,
    )
    return count_results


def _compute_q3_results(
    pair: SupervisedPair,
    pooling_type: PoolingType,
    projections: torch.Tensor,
    pooling_annotations_list: list[TextPairAnnotations],
    experiment_config: ExperimentConfig,
) -> Q3Results:
    valued_pairs_mask = ~projections.isnan()
    projections = projections[valued_pairs_mask].double()
    pooling_annotations_list = [
        annotations
        for annotations, is_valued
        in zip(pooling_annotations_list, valued_pairs_mask.tolist())
        if is_valued
    ]
    num_words = torch.tensor(
        [
            max(annotations['edit_data_by_threshold']['strict']['num_words_aae'], 1)
            for annotations in pooling_annotations_list
        ],
        dtype=torch.float64,
    )
    edit_counts_dict = _compute_edit_counts_dict(pooling_annotations_list, num_words, experiment_config.q3.edits_by_count)
    count_results = {
        edit_count: _compute_q3_count_results(
            projections, edit_counts, num_words.log(), experiment_config
        )
        for edit_count, edit_counts in edit_counts_dict.items()
    }
    results = Q3Results(
        corpus_name=pair.corpus_name,
        pooling_type=pooling_type,
        count_results=count_results,
    )
    return results


def compute_and_save_q3_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    primary_projections: dict[PoolingType, torch.Tensor],
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    q3_results = []
    for pooling_type in tqdm(experiment_config.q3_q4_pooling_types, desc='Q3 Poolings', leave=False):
        pooling_annotations_list = get_pooling_annotations_list(pair_annotations_list, pooling_type)
        q3_results.append(_compute_q3_results(
            pair, pooling_type, primary_projections[pooling_type], pooling_annotations_list, experiment_config,
        ))

    count_results_list = [
        count_results
        for results in q3_results
        for count_results in results['count_results'].values()
    ]
    for p_value_key in ('spearman_p_value', 'partial_spearman_p_value'): #
        p_values_fdr = false_discovery_control([
            count_results[p_value_key]
            for count_results in count_results_list
        ])
        for count_results, p_value in zip(count_results_list, p_values_fdr):
            count_results[f'{p_value_key}_fdr'] = float(p_value)
    save_json(q3_results, out_dir, f'{Q3_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}')
