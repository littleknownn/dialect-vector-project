import argparse
from pathlib import Path
from typing import NamedTuple

from scipy.stats import (
    false_discovery_control,
    spearmanr,
)
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.core.config import (
    add_model_argument,
    get_model_configs,
)
from src.core.config_utils import (
    DatasetConfig,
    ExperimentConfig,
    Q1LabeledThresholds,
    SupervisedPair,
    get_estimator_data,
)
from src.core.data_provenance import (
    save_freeze_record,
    verify_run_setup,
)
from src.core.paths import (
    FOLD_IDS_PREFIX,
    PROCESSED_DATA_DIR,
    Q1_PROJECTIONS_PREFIX,
    Q1_SHUFFLED_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q3_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    Q4_ROBUSTNESS_PREFIX,
    PAIR_EDIT_DATA_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
import src.core.edit_regression as edit_regression
import src.core.stats_utils as stats_utils
from src.core.utils import (
    CIStats,
    DifferenceVector,
    PairAnnotations,
    PairEditData,
    PoolingType,
    Q1ConsistencyLabel,
    Q1Results,
    Q3CountResults,
    Q3EditCount,
    Q3Results,
    Q4CurveFunction,
    Q4CurveResults,
    Q4Edit,
    Q4EditKind,
    Q4EditVerdict,
    Q4Results,
    Q4RobustnessResults,
    Q4RobustnessVariant,
    Q4Stratum,
    RunMode,
    SeedOffset,
    SplitSuffixes,
    TextPairAnnotations,
    load_json,
    save_json,
)


class Q1Inputs(NamedTuple):
    minuend_activation_matrix: torch.Tensor
    subtrahend_activation_matrix: torch.Tensor
    fold_ids: torch.Tensor
    mean_vector_folds: torch.Tensor


class Q4ParameterCIs(NamedTuple):
    weights: dict[Q4EditKind, CIStats]
    distance_decay_exponent: CIStats
    size_exponent: CIStats
    shares: dict[Q4EditKind, CIStats]
    weight_p_values: dict[Q4EditKind, float] | None
    num_failed_fits: int


class Q1Projections(NamedTuple):
    dialect_projections_by_pooling: dict[PoolingType, torch.Tensor]
    primary_pooling_projections_by_vector: dict[DifferenceVector, torch.Tensor]


class DoseBins(NamedTuple):
    bin_ids: torch.Tensor
    bin_edges: list[float]
    has_zero_bin: bool


### Pair Edit Data ###
def _load_pair_annotations_list(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> list[PairAnnotations]:
    pair_annotations_list = load_json(
        run_dirs.text_annotations / f'{pair.corpus_name}{SplitSuffixes.PAIR_ANNOTATIONS}{split_suffix}.json',
    )[:experiment_config.debug_max_split_lines]
    return pair_annotations_list


def _get_pooling_annotations_list(
    pair_annotations_list: list[PairAnnotations],
    pooling_type: PoolingType,
) -> list[TextPairAnnotations]:
    pooling_annotations_list = [
        pair_annotations['context']
        if pooling_type == PoolingType.BOUNDARY
        and pair_annotations['context'] is not None
        else pair_annotations['line']
        for pair_annotations in pair_annotations_list
    ]
    return pooling_annotations_list


def _get_identical_pairs_mask(
    pair_annotations_list: list[PairAnnotations],
    pooling_type: PoolingType,
) -> torch.Tensor:
    identical_pairs_mask = torch.tensor([
        annotations['edit_data_by_threshold']['strict']['stratum'] == Q4Stratum.IDENTICAL
        for annotations in _get_pooling_annotations_list(pair_annotations_list, pooling_type)
    ])
    return identical_pairs_mask


def _save_pair_edit_data(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    pair_annotations_list: list[PairAnnotations],
    out_dir: Path,
) -> list[PairEditData]:
    pair_edit_data = [
        pair_annotations['line']['edit_data_by_threshold']['strict']
        for pair_annotations in pair_annotations_list
    ]
    save_json(pair_edit_data, out_dir, f'{PAIR_EDIT_DATA_PREFIX}{pair.corpus_name}{split_suffix}')
    return pair_edit_data


### Q1 ###
def _load_q1_inputs_dict(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    pooling_type: PoolingType,
    layer: int,
    vectors: tuple[DifferenceVector, ...],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path,
) -> dict[DifferenceVector, Q1Inputs]:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    fold_ids = torch.load(
        run_dirs.fold_vectors / f'{FOLD_IDS_PREFIX}{pair.corpus_name}{split_suffix}.pt',
        weights_only=True,
    )
    load_activation_matrix_fn = lambda dialect_suffix: torch.load(
        get_sweep_tensor_filepath(
            run_dirs.activations,
            sweep_identifier,
            f'{pair.corpus_name}{dialect_suffix}{split_suffix}',
        ),
        weights_only=True,
    ).float()

    aae_activation_matrix = load_activation_matrix_fn(SplitSuffixes.AAE)
    sae_activation_matrix = load_activation_matrix_fn(SplitSuffixes.SAE)
    aae_back_activation_matrix = None
    if any(vector.includes_aae_back for vector in vectors):
        pair_metadata = load_json(
            processed_data_dir / f'{pair.corpus_name}{SplitSuffixes.PAIR_METADATA}{split_suffix}.json',
        )[:experiment_config.debug_max_split_lines]
        aae_back_invalid_mask = torch.tensor([
            not metadata['is_aae_back_valid']
            for metadata in pair_metadata
        ])
        aae_back_activation_matrix = load_activation_matrix_fn(SplitSuffixes.AAE_BACK)
        aae_back_activation_matrix[aae_back_invalid_mask] = float('nan')
    
    q1_inputs_dict = {}
    for vector in vectors:
        mean_vector_folds = torch.load(
            get_sweep_tensor_filepath(
                run_dirs.fold_vectors,
                sweep_identifier,
                f'{pair.corpus_name}{vector}{split_suffix}',
            ),
            weights_only=True,
        )
        minuend_activation_matrix = (
            aae_back_activation_matrix
            if vector == DifferenceVector.DIALECT_GEMINI
            else aae_activation_matrix
        )
        subtrahend_activation_matrix = (
            aae_back_activation_matrix
            if vector == DifferenceVector.AUTHORSHIP
            else sae_activation_matrix
        )
        q1_inputs_dict[vector] = Q1Inputs(
            minuend_activation_matrix,
            subtrahend_activation_matrix,
            fold_ids,
            mean_vector_folds,
        )
    
    return q1_inputs_dict


def _get_q1_label(
    positive_projection_ratio: float,
    standardized_mean_ci_lower: float,
    thresholds: Q1LabeledThresholds,
) -> Q1ConsistencyLabel:
    if (
        positive_projection_ratio >= thresholds.consistent.positive_projection_ratio
        and standardized_mean_ci_lower > thresholds.consistent.standardized_mean_ci_lower
    ):
        return Q1ConsistencyLabel.CONSISTENT
    if (
        positive_projection_ratio < thresholds.weak.positive_projection_ratio
        or standardized_mean_ci_lower <= thresholds.weak.standardized_mean_ci_lower
    ):
        return Q1ConsistencyLabel.WEAK
    return Q1ConsistencyLabel.PARTIAL


def _compute_q1_results(
    projections: torch.Tensor,
    q1_inputs: Q1Inputs,
    identical_pairs_mask: torch.Tensor,
    ending_changed_mask: torch.Tensor,
    difference_vector_cosine: float,
    pair: SupervisedPair,
    vector: DifferenceVector,
    layer: int,
    pooling_type: PoolingType,
    experiment_config: ExperimentConfig,
    shuffled_pairing_idx: int = 0,
) -> tuple[Q1Results, torch.Tensor]:
    generator = torch.Generator().manual_seed(experiment_config.random_seed + SeedOffset.SHUFFLED_PAIRING)
    valued_pairs_mask = ~projections.isnan()
    num_excluded_pairs = len(projections) - int(valued_pairs_mask.sum())

    projections = projections[valued_pairs_mask]
    identical_pairs_mask = identical_pairs_mask[valued_pairs_mask]
    ending_changed_mask = ending_changed_mask[valued_pairs_mask]

    bootstrap_fn = lambda values, statistic=None: stats_utils.compute_bootstrap_ci(
        values,
        statistic,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + SeedOffset.BOOTSTRAP,
    )
    projection_mean_ci = bootstrap_fn(projections)
    standardized_projection_mean_ci = bootstrap_fn(projections, stats_utils.compute_standardized_projection_mean)
    positive_projection_ratio_ci = bootstrap_fn(projections, stats_utils.compute_positive_projection_ratio)
    ending_same_projection_mean_ci = bootstrap_fn(projections[~ending_changed_mask])
    ending_changed_projection_mean_ci = bootstrap_fn(projections[ending_changed_mask])

    minuend_projections = stats_utils.compute_fold_projections(
        q1_inputs.minuend_activation_matrix,
        q1_inputs.fold_ids,
        q1_inputs.mean_vector_folds,
    )[valued_pairs_mask].double()
    unit_vector_folds = F.normalize(q1_inputs.mean_vector_folds, dim=0)
    subtrahend_projections_by_fold = (
        q1_inputs.subtrahend_activation_matrix @ unit_vector_folds
    )[valued_pairs_mask].double()
    fold_ids = q1_inputs.fold_ids[valued_pairs_mask]

    shuffled_projections = torch.stack([
        minuend_projections - subtrahend_projections_by_fold[
            torch.randperm(len(minuend_projections), generator=generator),
            fold_ids,
        ]
        for _ in range(experiment_config.q1_num_shuffled_pairings)
    ])
    differences = (
        q1_inputs.minuend_activation_matrix - q1_inputs.subtrahend_activation_matrix
    )[valued_pairs_mask].double()

    nonidentical_projections = projections[~identical_pairs_mask]
    nonidentical_pairs_positive_projection_ratio_ci = bootstrap_fn(
        nonidentical_projections, stats_utils.compute_positive_projection_ratio,
    )
    nonidentical_pairs_standardized_projection_mean_ci = bootstrap_fn(
        nonidentical_projections, stats_utils.compute_standardized_projection_mean,
    )
    label = _get_q1_label(
        nonidentical_pairs_positive_projection_ratio_ci['estimate'],
        nonidentical_pairs_standardized_projection_mean_ci['ci_lower'],
        experiment_config.q1_labeled_bounds,
    )

    mean_squared_norm = torch.linalg.vector_norm(differences, dim=1).square().mean()
    activation_difference_energy = float(torch.linalg.vector_norm(differences.mean(dim=0)).square() / mean_squared_norm)
    projection_energy = float(projections.double().square().mean() / mean_squared_norm)
    content_matching_gain = float(
        1 - projections.double().var() / shuffled_projections.var(dim=1).mean()
    )

    sign_flip_p_value = stats_utils.compute_sign_flip_p_value(projections, experiment_config.q1_num_sign_flips, experiment_config.random_seed + SeedOffset.SIGN_FLIP)
    
    identical_pairs_median_abs_projection = float(projections[identical_pairs_mask].abs().median())
    other_pairs_median_abs_projection = float(projections[~identical_pairs_mask].abs().median())
    
    shuffled_positive_projection_ratio = float(stats_utils.compute_positive_projection_ratio(shuffled_projections).mean())
    shuffled_standardized_projection_mean = float(stats_utils.compute_standardized_projection_mean(shuffled_projections).mean())

    results = Q1Results(
        corpus_name=pair.corpus_name,
        label=label,
        vector=vector,
        layer=layer,
        pooling_type=pooling_type,
        num_pairs=len(projections),
        num_excluded_pairs=num_excluded_pairs,

        projection_mean_ci=projection_mean_ci,
        standardized_projection_mean_ci=standardized_projection_mean_ci,
        positive_projection_ratio_ci=positive_projection_ratio_ci,
        ending_same_projection_mean_ci=ending_same_projection_mean_ci,
        ending_changed_projection_mean_ci=ending_changed_projection_mean_ci,

        activation_difference_energy=activation_difference_energy,
        projection_energy=projection_energy,
        content_matching_gain=content_matching_gain,

        sign_flip_p_value=sign_flip_p_value,
        sign_flip_p_value_fdr=None,
        
        identical_pairs_median_abs_projection=identical_pairs_median_abs_projection,
        other_pairs_median_abs_projection=other_pairs_median_abs_projection,

        shuffled_positive_projection_ratio=shuffled_positive_projection_ratio,
        shuffled_standardized_projection_mean=shuffled_standardized_projection_mean,

        num_identical_pairs=int(identical_pairs_mask.sum()),
        nonidentical_pairs_positive_projection_ratio_ci=nonidentical_pairs_positive_projection_ratio_ci,
        nonidentical_pairs_standardized_projection_mean_ci=nonidentical_pairs_standardized_projection_mean_ci,
        difference_vector_cosine=difference_vector_cosine,
    )
    return results, shuffled_projections[shuffled_pairing_idx].float()


def _compute_and_save_q1_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    pair_edit_data: list[PairEditData],
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path,
) -> Q1Projections:
    ending_changed_mask = torch.tensor([
        edit_data['ending_changed']
        for edit_data in pair_edit_data
    ])
    
    q1_results = []
    q1_projections = Q1Projections(dialect_projections_by_pooling={}, primary_pooling_projections_by_vector={})
    for layer in experiment_config.target_layers:
        vectors = (
            tuple(DifferenceVector)
            if layer == experiment_config.primary_layer
            else experiment_config.q1_layer_sweep_vectors
        )
        for pooling_type in PoolingType:
            identical_pairs_mask = _get_identical_pairs_mask(pair_annotations_list, pooling_type)
            q1_inputs_dict = _load_q1_inputs_dict(
                pair, split_suffix, pooling_type, layer, vectors, experiment_config, run_dirs, processed_data_dir,
            )
            difference_vector = q1_inputs_dict[DifferenceVector.DIALECT].mean_vector_folds.mean(dim=1)
            for vector, q1_inputs in q1_inputs_dict.items():
                projections = stats_utils.compute_fold_projections(
                    q1_inputs.minuend_activation_matrix - q1_inputs.subtrahend_activation_matrix,
                    q1_inputs.fold_ids,
                    q1_inputs.mean_vector_folds,
                )
                difference_vector_cosine = float(F.cosine_similarity(q1_inputs.mean_vector_folds.mean(dim=1), difference_vector, dim=0))
                results, shuffled_projections = _compute_q1_results(
                    projections, q1_inputs, identical_pairs_mask, ending_changed_mask, difference_vector_cosine, pair, vector, layer, pooling_type, experiment_config,
                )
                q1_results.append(results)
                if layer != experiment_config.primary_layer:
                    continue
                if vector == DifferenceVector.DIALECT:
                    q1_projections.dialect_projections_by_pooling[pooling_type] = projections
                    sweep_identifier = get_sweep_identifier(layer, pooling_type)
                    torch.save(
                        shuffled_projections,
                        run_dirs.pair_results / f'{Q1_SHUFFLED_PROJECTIONS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}.pt',
                    )
                if pooling_type == experiment_config.primary_pooling_type:
                    q1_projections.primary_pooling_projections_by_vector[vector] = projections
    
    primary_results = [
        results for results in q1_results
        if results['layer'] == experiment_config.primary_layer
        and results['vector'] in experiment_config.q1_fdr_vectors
    ]
    p_values_fdr = false_discovery_control([
        results['sign_flip_p_value'] for results in primary_results
    ])
    for results, p_value in zip(primary_results, p_values_fdr):
        results['sign_flip_p_value_fdr'] = float(p_value)
    
    for pooling_type, projections in q1_projections.dialect_projections_by_pooling.items():
        sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
        torch.save(projections, run_dirs.pair_results / f'{Q1_PROJECTIONS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}.pt')
    save_json(q1_results, run_dirs.pair_results, f'{Q1_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}')
    return q1_projections


### Q3 ###
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
    bins = _get_edit_count_bins(edit_counts, experiment_config.q3_num_dose_bins)

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
    edit_counts_dict = _compute_edit_counts_dict(pooling_annotations_list, num_words, experiment_config.q3_edits_by_count)
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


def _compute_and_save_q3_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    primary_projections: dict[PoolingType, torch.Tensor],
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    q3_results = []
    for pooling_type in tqdm(experiment_config.q3_q4_pooling_types, desc='Q3 Poolings', leave=False):
        pooling_annotations_list = _get_pooling_annotations_list(pair_annotations_list, pooling_type)
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


### Q4 ###
def _compute_q4_parameter_cis(
    fit: edit_regression.Fit,
    regression_inputs: edit_regression.Inputs,
    num_draws: int,
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
    reference_weight: float = 1.0,
) -> Q4ParameterCIs:
    parameters = edit_regression.unpack_parameters(fit, regression_inputs.controls.shape[1])
    regression_shares = edit_regression.compute_shares(fit, regression_inputs)
    draws = edit_regression.bootstrap_regression(
        fit, regression_inputs, num_draws, experiment_config.random_seed + SeedOffset.Q3_Q4_BOOTSTRAP,
    )
    ci_fn = lambda parameter_draws, estimate: stats_utils.compute_percentile_ci(
        parameter_draws, experiment_config.p_alpha, float(estimate),
    )
    weights = {
        kind: ci_fn(draws.weights[:, kind_idx], parameters.weights[kind_idx])
        for kind_idx, kind in enumerate(edit_kinds)
    }
    distance_decay_exponent = ci_fn(draws.distance_decay_exponents, parameters.distance_decay_exponent)
    size_exponent = ci_fn(draws.size_exponents, parameters.size_exponent)
    shares = {
        kind: ci_fn(draws.kind_shares[:, kind_idx], regression_shares.by_kind[kind_idx])
        for kind_idx, kind in enumerate(edit_kinds)
    }
    weight_p_values = {
        kind: stats_utils.compute_bootstrap_p_value(draws.weights[:, kind_idx], reference_weight)
        for kind_idx, kind in enumerate(edit_kinds)
        if kind != Q4EditKind.CONTENT_CHANGE
    }
    parameter_cis = Q4ParameterCIs(
        weights=weights,
        distance_decay_exponent=distance_decay_exponent,
        size_exponent=size_exponent,
        shares=shares,
        weight_p_values=weight_p_values,
        num_failed_fits=draws.num_failed_fits,
    )
    return parameter_cis


def _get_q4_parameter_estimates(
    fit: edit_regression.Fit,
    regression_inputs: edit_regression.Inputs,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4ParameterCIs:
    parameters = edit_regression.unpack_parameters(fit, regression_inputs.controls.shape[1])
    regression_shares = edit_regression.compute_shares(fit, regression_inputs)
    point_fn = lambda estimate: CIStats(
        estimate=float(estimate),
        ci_lower=float('nan'),
        ci_upper=float('nan'),
    )
    weights = {
        kind: point_fn(parameters.weights[kind_idx])
        for kind_idx, kind in enumerate(edit_kinds)
    }
    distance_decay_exponent = point_fn(parameters.distance_decay_exponent)
    size_exponent = point_fn(parameters.size_exponent)
    shares = {
        kind: point_fn(regression_shares.by_kind[kind_idx])
        for kind_idx, kind in enumerate(edit_kinds)
    }
    parameter_estimates = Q4ParameterCIs(
        weights=weights,
        distance_decay_exponent=distance_decay_exponent,
        size_exponent=size_exponent,
        shares=shares,
        weight_p_values=None,
        num_failed_fits=0,
    )
    return parameter_estimates


def _correct_weight_p_values(
    weight_p_values: dict[Q4EditKind, float],
    tested_edit_kinds: tuple[Q4EditKind, ...],
) -> dict[Q4EditKind, float]:
    if not tested_edit_kinds:
        return {}
    p_values_fdr = false_discovery_control([weight_p_values[kind] for kind in tested_edit_kinds])
    weight_p_values_fdr = {
        kind: float(p_value)
        for kind, p_value in zip(tested_edit_kinds, p_values_fdr)
    }
    return weight_p_values_fdr


def _compute_q4_curve_results(
    regression_inputs: edit_regression.Inputs,
    curve: Q4CurveFunction,
    bootstrap: bool,
    tested_edit_kinds: tuple[Q4EditKind, ...],
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4CurveResults:
    fit = edit_regression.fit_regression(regression_inputs, curve, experiment_config.q4_regression_settings)
    predictions = edit_regression.predict(fit, regression_inputs).projections
    held_out_predictions = edit_regression.compute_held_out_predictions(
        fit, regression_inputs, experiment_config.q4_num_held_out_folds, experiment_config.random_seed + SeedOffset.Q4_HELD_OUT_FOLDS,
    )
    regression_shares = edit_regression.compute_shares(fit, regression_inputs)
    parameter_cis = (
        _compute_q4_parameter_cis(fit, regression_inputs, experiment_config.q4_num_bootstrap_samples, experiment_config)
        if bootstrap
        else _get_q4_parameter_estimates(fit, regression_inputs)
    )
    distances = (
        experiment_config.q4_marginal_effect_distances
        if regression_inputs.uses_distance_decay
        else (1,)
    )
    by_kind_fn = lambda values:{
        kind: float(value)
        for kind, value
        in zip(edit_kinds, values)
    }
    mean_marginal_effects_by_distance = {
        distance: by_kind_fn(edit_regression.compute_mean_marginal_effects(
            fit, regression_inputs, distance,
        ))
        for distance in distances
    }

    r_squared = float(stats_utils.compute_r_squared(predictions, regression_inputs.projections))
    held_out_r_squared = float(stats_utils.compute_r_squared(held_out_predictions, regression_inputs.projections))
    distance_decay_exponent = (
        parameter_cis.distance_decay_exponent
        if regression_inputs.uses_distance_decay
        else None
    )
    calibration = stats_utils.compute_calibration(
        predictions, regression_inputs.projections, experiment_config.q4_num_calibration_bins,
    )
    mean_residual_by_kind = by_kind_fn(edit_regression.compute_mean_residual_by_kind(predictions, regression_inputs))
    shuffled_r_squared = edit_regression.compute_shuffled_r_squared(
        fit, regression_inputs, experiment_config.random_seed + SeedOffset.Q4_SHUFFLED_PROJECTIONS,
    )
    weight_p_values_fdr = (
        _correct_weight_p_values(parameter_cis.weight_p_values, tested_edit_kinds)
        if parameter_cis.weight_p_values is not None
        else None
    )
    curve_results = Q4CurveResults(
        curve=curve,
        r_squared=r_squared,
        held_out_r_squared=held_out_r_squared,
        distance_decay_exponent=distance_decay_exponent,
        size_exponent=parameter_cis.size_exponent,
        weights=parameter_cis.weights,
        mean_marginal_effects_by_distance=mean_marginal_effects_by_distance,
        shares=parameter_cis.shares,
        format_flags_share=regression_shares.format_flags,
        pair_controls_share=regression_shares.pair_controls,
        unexplained_share=regression_shares.unexplained,
        calibration=calibration,
        mean_residual_by_kind=mean_residual_by_kind,
        shuffled_r_squared=shuffled_r_squared,
        weight_p_values=parameter_cis.weight_p_values,
        weight_p_values_fdr=weight_p_values_fdr,
        num_failed_fits=parameter_cis.num_failed_fits,
    )
    return curve_results


def _get_q4_tested_edit_kinds(
    num_edits_by_kind: dict[Q4EditKind, int],
    min_edits_for_verdict: int,
) -> tuple[Q4EditKind, ...]:
    tested_edit_kinds = tuple(
        edit_kind
        for edit_kind, num_edits in num_edits_by_kind.items()
        if edit_kind != Q4EditKind.CONTENT_CHANGE
        and num_edits >= min_edits_for_verdict
    )
    return tested_edit_kinds


def _get_q4_verdict(
    edit_kind: Q4EditKind,
    curve_results_list: list[Q4CurveResults],
    p_alpha: float,
    reference_weight: float = 1.0,
) -> Q4EditVerdict:
    if not all(
        curve_results['weight_p_values_fdr'][edit_kind] < p_alpha
        for curve_results in curve_results_list
    ):
        return Q4EditVerdict.NOT_DIFFERENT
    if all(
        curve_results['weights'][edit_kind]['estimate'] > reference_weight
        for curve_results in curve_results_list
    ):
        return Q4EditVerdict.STRONGER
    if all(
        curve_results['weights'][edit_kind]['estimate'] < reference_weight
        for curve_results in curve_results_list
    ):
        return Q4EditVerdict.WEAKER
    return Q4EditVerdict.NOT_DIFFERENT


def _get_q4_verdicts(
    curve_results: dict[Q4CurveFunction, Q4CurveResults],
    tested_edit_kinds: tuple[Q4EditKind, ...],
    experiment_config: ExperimentConfig,
) -> dict[Q4EditKind, Q4EditVerdict]:
    primary_curve_results = [curve_results[curve] for curve in experiment_config.q4_primary_curves]
    verdicts = {
        edit_kind: (
            _get_q4_verdict(edit_kind, primary_curve_results, experiment_config.p_alpha)
            if edit_kind in tested_edit_kinds
            else Q4EditVerdict.TOO_RARE
        )
        for edit_kind in Q4EditKind
        if edit_kind != Q4EditKind.CONTENT_CHANGE
    }
    return verdicts


def _compute_mean_projection_by_stratum(
    projections: torch.Tensor,
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
) -> tuple[dict[Q4Stratum, int], dict[Q4Stratum, CIStats]]:
    strata = [edit_data['stratum'] for edit_data in pair_edit_data]
    num_pairs_by_stratum = {}
    mean_projection_by_stratum = {}
    for stratum in Q4Stratum:
        stratum_mask = torch.tensor([pair_stratum == stratum for pair_stratum in strata])
        num_pairs_by_stratum[stratum] = int(stratum_mask.sum())
        if not stratum_mask.any():
            continue
        mean_projection_by_stratum[stratum] = stats_utils.compute_bootstrap_ci(
            values=projections[stratum_mask],
            num_draws=experiment_config.num_bootstrap_samples,
            p_alpha=experiment_config.p_alpha,
            seed=experiment_config.random_seed + SeedOffset.BOOTSTRAP,
        )
    return num_pairs_by_stratum, mean_projection_by_stratum


def _compute_ceiling_minus_log_r_squared(
    values: torch.Tensor,
) -> torch.Tensor:
    targets, log_predictions, ceiling_predictions = values.unbind(dim=-1)
    ceiling_minus_log_r_squared = (
        stats_utils.compute_r_squared(ceiling_predictions, targets)
        - stats_utils.compute_r_squared(log_predictions, targets)
    )
    return ceiling_minus_log_r_squared


def _compute_ceiling_minus_log_held_out_r_squared(
    regression_inputs: edit_regression.Inputs,
    experiment_config: ExperimentConfig,
) -> CIStats:
    held_out_predictions_fn = lambda curve: edit_regression.compute_held_out_predictions(
        edit_regression.fit_regression(
            regression_inputs, curve, experiment_config.q4_regression_settings,
        ),
        regression_inputs,
        experiment_config.q4_num_held_out_folds,
        experiment_config.random_seed + SeedOffset.Q4_HELD_OUT_FOLDS,
    )
    values = torch.stack(
        [
            regression_inputs.projections,
            held_out_predictions_fn(Q4CurveFunction.LOG),
            held_out_predictions_fn(Q4CurveFunction.CEILING),
        ],
        dim=1,
    )
    ceiling_minus_log_held_out_r_squared = stats_utils.compute_bootstrap_ci(
        values,
        _compute_ceiling_minus_log_r_squared,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + SeedOffset.Q3_Q4_BOOTSTRAP,
    )
    return ceiling_minus_log_held_out_r_squared


def _build_q4_regression_inputs(
    projections: torch.Tensor,
    pooling_annotations_list: list[TextPairAnnotations],
    pair_edits_list: list[list[Q4Edit]],
    pooling_type: PoolingType,
    experiment_config: ExperimentConfig,
) -> edit_regression.Inputs:
    valued_pairs_mask = ~projections.isnan()
    valued_pair_indices = valued_pairs_mask.nonzero().squeeze(1).tolist()
    regression_inputs = edit_regression.build_regression_inputs(
        projections[valued_pairs_mask],
        pair_edits_list=[
            pair_edits_list[pair_idx] for pair_idx in valued_pair_indices
        ],
        pair_edit_data_list=[
            pooling_annotations_list[pair_idx]['edit_data_by_threshold']['strict']
            for pair_idx in valued_pair_indices
        ],
        pair_controls_list=[
            pooling_annotations_list[pair_idx]['controls']
            for pair_idx in valued_pair_indices
        ],
        format_flags=experiment_config.q4_format_flags,
        uses_distance_decay=(pooling_type == PoolingType.BOUNDARY),
    )
    return regression_inputs


def _compute_q4_results(
    pair: SupervisedPair,
    pooling_type: PoolingType,
    projections: torch.Tensor,
    pooling_annotations_list: list[TextPairAnnotations],
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4Results:
    valued_pairs_mask = ~projections.isnan()
    pair_edit_data_list = [
        annotations['edit_data_by_threshold']['strict']
        for annotations, is_valued
        in zip(pooling_annotations_list, valued_pairs_mask.tolist())
        if is_valued
    ]
    regression_inputs = _build_q4_regression_inputs(
        projections,
        pooling_annotations_list,
        [annotations['edits'] for annotations in pooling_annotations_list],
        pooling_type,
        experiment_config,
    )
    is_bootstrap = pooling_type in experiment_config.q3_q4_pooling_types
    num_pairs_by_stratum, mean_projection_by_stratum = _compute_mean_projection_by_stratum(
        regression_inputs.projections, pair_edit_data_list, experiment_config,
    )
    num_edits_by_kind = {
        edit_kind: int(num_edits)
        for edit_kind, num_edits
        in zip(edit_kinds, edit_regression.count_edits_by_kind(regression_inputs))
    }
    tested_edit_kinds = _get_q4_tested_edit_kinds(num_edits_by_kind, experiment_config.q4_min_edits_for_verdict)
    curve_results = {
        curve:_compute_q4_curve_results(
            regression_inputs,
            curve,
            is_bootstrap and curve in experiment_config.q4_primary_curves,
            tested_edit_kinds,
            experiment_config,
        )
        for curve in Q4CurveFunction
    }
    verdicts = (
        _get_q4_verdicts(curve_results, tested_edit_kinds, experiment_config)
        if is_bootstrap
        else None
    )
    single_edit_weights = (
        edit_regression.compute_single_edit_weight_cis(
            regression_inputs,
            experiment_config.q4_num_bootstrap_samples,
            experiment_config.p_alpha,
            experiment_config.random_seed + SeedOffset.Q3_Q4_BOOTSTRAP,
        )
        if regression_inputs.uses_distance_decay
        else None
    )
    ceiling_minus_log_held_out_r_squared = _compute_ceiling_minus_log_held_out_r_squared(regression_inputs, experiment_config)
    results = Q4Results(
        corpus_name=pair.corpus_name,
        pooling_type=pooling_type,
        num_pairs_by_stratum=num_pairs_by_stratum,
        mean_projection_by_stratum=mean_projection_by_stratum,
        num_edits_by_kind=num_edits_by_kind,
        curve_results=curve_results,
        ceiling_minus_log_held_out_r_squared=ceiling_minus_log_held_out_r_squared,
        verdicts=verdicts,
        single_edit_weights=single_edit_weights,
    )
    return results


def _compute_and_save_q4_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    primary_projections: dict[PoolingType, torch.Tensor],
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    for pooling_type in experiment_config.q3_q4_pooling_types:
        sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
        pooling_annotations_list = _get_pooling_annotations_list(pair_annotations_list, pooling_type)
        q4_results = _compute_q4_results(pair, pooling_type, primary_projections[pooling_type], pooling_annotations_list, experiment_config)
        save_json(q4_results, out_dir, f'{Q4_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}')


### Q4 Robustness ###
_SINGLE_WORD_EDIT_SIZE: int = 1
_VECTOR_BY_ROBUSTNESS_VARIANT: dict[Q4RobustnessVariant, DifferenceVector] = {
    Q4RobustnessVariant.NO_PROFANITY_VECTOR: DifferenceVector.NO_PROFANITY,
    Q4RobustnessVariant.NO_TYPOGRAPHY_VECTOR: DifferenceVector.NO_TYPOGRAPHY,
    Q4RobustnessVariant.SAME_TERMINAL_PUNCTUATION_VECTOR: DifferenceVector.SAME_TERMINAL_PUNCTUATION,
}


class Q4RobustnessInputs(NamedTuple):
    variant: Q4RobustnessVariant
    removed_word: str | None
    regression_inputs: edit_regression.Inputs


def _build_q4_robustness_inputs_list(
    q1_projections: Q1Projections,
    pooling_annotations_list: list[TextPairAnnotations],
    experiment_config: ExperimentConfig,
    single_word_edit_size: int = _SINGLE_WORD_EDIT_SIZE,
    vector_by_robustness_variant: dict[Q4RobustnessVariant, DifferenceVector] = _VECTOR_BY_ROBUSTNESS_VARIANT,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> list[Q4RobustnessInputs]:
    build_fn = lambda projections, pair_edits_list: _build_q4_regression_inputs(
        projections, pooling_annotations_list, pair_edits_list, experiment_config.primary_pooling_type, experiment_config,
    )
    relabeled_build_fn = lambda kind_fn: build_fn(
        q1_projections.primary_pooling_projections_by_vector[DifferenceVector.DIALECT],
        [
            [
                edit | {'kind': kind_fn(edit)}
                for edit in annotations['edits']
            ]
            for annotations in pooling_annotations_list
        ],
    )

    pair_edits_list = [annotations['edits'] for annotations in pooling_annotations_list]
    primary_inputs = build_fn(
        q1_projections.primary_pooling_projections_by_vector[DifferenceVector.DIALECT],
        pair_edits_list,
    )
    multi_word_marker_mask = (
        (primary_inputs.edit_kind_indices == edit_kinds.index(Q4EditKind.GRAMMAR_WORD))
        & (primary_inputs.edit_sizes > single_word_edit_size)
    )
    robustness_inputs_list = [
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.SINGLE_WORD_MARKERS,
            removed_word=None,
            regression_inputs=edit_regression.relabel_edit_kinds(
                primary_inputs, multi_word_marker_mask, Q4EditKind.CONTENT_CHANGE,
            ),
        ),
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.AINT_EXCLUDED,
            removed_word=None,
            regression_inputs=relabeled_build_fn(
                lambda edit: edit['kind_without_aint']
            ),
        ),
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.DISTANCE_FROM_TWO,
            removed_word=None,
            regression_inputs=primary_inputs._replace(
                edit_distances=primary_inputs.edit_distances + experiment_config.q4_robustness_distance_shift,
            ),
        ),
    ]
    robustness_inputs_list += [
        Q4RobustnessInputs(variant, removed_word=None, regression_inputs=build_fn(
            q1_projections.primary_pooling_projections_by_vector[vector],
            pair_edits_list,
        ))
        for variant, vector in vector_by_robustness_variant.items()
    ]
    present_marker_words = sorted({
        word
        for pair_edits in pair_edits_list
        for edit in pair_edits
        for word in edit['kind_by_removed_marker_word']
    })
    robustness_inputs_list += [
        Q4RobustnessInputs(
            variant=Q4RobustnessVariant.LEAVE_ONE_OUT,
            removed_word=word,
            regression_inputs=relabeled_build_fn(
                lambda edit:
                edit['kind_by_removed_marker_word'].get(word, edit['kind']),
            )
        )
        for word in present_marker_words
    ]
    return robustness_inputs_list


def _compute_q4_robustness_results(
    pair: SupervisedPair,
    robustness_inputs: Q4RobustnessInputs,
    curve: Q4CurveFunction,
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4RobustnessResults:
    fit = edit_regression.fit_regression(
        robustness_inputs.regression_inputs, curve, experiment_config.q4_regression_settings,
    )
    parameter_cis = (
        _get_q4_parameter_estimates(fit, robustness_inputs.regression_inputs)
        if robustness_inputs.variant == Q4RobustnessVariant.LEAVE_ONE_OUT
        else _compute_q4_parameter_cis(
            fit, robustness_inputs.regression_inputs, experiment_config.q4_num_robustness_bootstrap_samples, experiment_config,
        )
    )
    num_edits_by_kind = edit_regression.count_edits_by_kind(robustness_inputs.regression_inputs)
    num_marker_edits = int(num_edits_by_kind[edit_kinds.index(Q4EditKind.GRAMMAR_WORD)])
    results = Q4RobustnessResults(
        corpus_name=pair.corpus_name,
        pooling_type=experiment_config.primary_pooling_type,
        variant=robustness_inputs.variant,
        removed_word=robustness_inputs.removed_word,
        curve=curve,
        num_marker_edits=num_marker_edits,
        weights=parameter_cis.weights,
    )
    return results


def _compute_and_save_q4_robustness_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    q1_projections: Q1Projections,
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    pooling_annotations_list = _get_pooling_annotations_list(pair_annotations_list, experiment_config.primary_pooling_type)
    robustness_inputs_list = _build_q4_robustness_inputs_list(q1_projections, pooling_annotations_list, experiment_config)
    robustness_results = [
        _compute_q4_robustness_results(pair, robustness_inputs, curve, experiment_config)
        for robustness_inputs in tqdm(robustness_inputs_list, desc='Q4 Robustness', leave=False)
        for curve in experiment_config.q4_primary_curves
    ]
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, experiment_config.primary_pooling_type)
    save_json(robustness_results, out_dir, f'{Q4_ROBUSTNESS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}')


### Main ###
def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    split_suffix: SplitSuffixes,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    num_steps_per_pair = 4
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    step_bar = tqdm(total=num_steps_per_pair * len(supervised_pairs), desc='06 Analyzing Pairs')
    for pair in supervised_pairs:
        pair_annotations_list = _load_pair_annotations_list(pair, split_suffix, experiment_config, run_dirs)
        pair_edit_data = _save_pair_edit_data(pair, split_suffix, pair_annotations_list, run_dirs.pair_results)
        
        step_bar.set_postfix_str(f'Q1 {pair.corpus_name}')
        q1_projections = _compute_and_save_q1_results(pair, split_suffix, pair_edit_data, pair_annotations_list, experiment_config, run_dirs, processed_data_dir)
        step_bar.update()
        
        step_bar.set_postfix_str(f'Q3 {pair.corpus_name}')
        _compute_and_save_q3_results(pair, split_suffix, q1_projections.dialect_projections_by_pooling, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()

        step_bar.set_postfix_str(f'Q4 {pair.corpus_name}')
        _compute_and_save_q4_results(pair, split_suffix, q1_projections.dialect_projections_by_pooling, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()

        step_bar.set_postfix_str(f'Q4 Robustness {pair.corpus_name}')
        _compute_and_save_q4_robustness_results(pair, split_suffix, q1_projections, pair_annotations_list, experiment_config, run_dirs.pair_results)
        step_bar.update()
    step_bar.close()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--mode', type=RunMode, choices=list(RunMode), default=RunMode.DEVELOPMENT,
    )
    add_model_argument(argument_parser)
    arguments = argument_parser.parse_args()
    corpus_config, experiment_config = get_model_configs(arguments.model)
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(experiment_config)
    verify_run_setup(experiment_config, run_dirs, arguments.mode)
    compute_and_save_metrics(corpus_config, arguments.mode.split_suffix, experiment_config, run_dirs)
    if arguments.mode == RunMode.FREEZE:
        save_freeze_record(experiment_config, run_dirs)
    tqdm.write('06: Pair Metrics Computation Complete\n')
