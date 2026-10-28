from pathlib import Path
from typing import NamedTuple

from scipy.stats import (
    false_discovery_control,
    spearmanr,
)
from tqdm import tqdm
import torch

from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
)
from src.core.config_utils import (
    DatasetConfig,
    ExperimentConfig,
    Q1LabeledThresholds,
    SupervisedPair,
    get_estimator_data,
)
from src.core.paths import (
    PROCESSED_DATA_DIR,
    Q1_FOLD_IDS_PREFIX,
    Q1_FOLD_MEAN_VECTORS_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q3_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    PAIR_EDIT_DATA_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
    verify_extraction_log,
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
    Q4EditKind,
    Q4EditVerdict,
    Q4Results,
    Q4Stratum,
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
def _load_fold_estimates(
    pair: SupervisedPair,
    vector: DifferenceVector,
    sweep_identifier: str,
    run_dirs: RunDirs,
) -> tuple[torch.Tensor, torch.Tensor]:
    fold_ids = torch.load(
        run_dirs.q1_estimators / f'{Q1_FOLD_IDS_PREFIX}{pair.corpus_name}.pt',
        weights_only=True,
    )
    mean_vector_folds = torch.load(
        run_dirs.q1_estimators / f'{Q1_FOLD_MEAN_VECTORS_PREFIX}{pair.corpus_name}{vector}_{sweep_identifier}.pt',
        weights_only=True,
        map_location='cpu', #
    )
    return fold_ids, mean_vector_folds


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
        fold_ids, mean_vector_folds = _load_fold_estimates(
            pair, vector, sweep_identifier, split_suffix, len(aae_activation_matrix), run_dirs,
        )
        minuend_activation_matrix = (
            aae_back_activation_matrix
            if vector == DifferenceVector.DIALECT
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
    pair: SupervisedPair,
    vector: DifferenceVector,
    layer: int,
    pooling_type: PoolingType,
    experiment_config: ExperimentConfig,
) -> Q1Results:    
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

    project_fn = lambda matrix: stats_utils.compute_fold_projections(
        matrix,
        q1_inputs.fold_ids,
        q1_inputs.mean_vector_folds
    )[valued_pairs_mask].double()
    minuend_projections = project_fn(q1_inputs.minuend_activation_matrix)
    subtrahend_projections = project_fn(q1_inputs.subtrahend_activation_matrix)

    shuffled_projections = torch.stack([
        minuend_projections - subtrahend_projections[
            torch.randperm(len(subtrahend_projections), generator=generator)
        ]
        for _ in range(experiment_config.q1_num_shuffled_pairings)
    ])
    differences = (
        q1_inputs.minuend_activation_matrix - q1_inputs.subtrahend_activation_matrix
    )[valued_pairs_mask].double()

    label = _get_q1_label(
        positive_projection_ratio_ci['estimate'],
        standardized_projection_mean_ci['ci_lower'],
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
    )
    return results


def _compute_and_save_q1_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path,
) -> dict[PoolingType, torch.Tensor]:
    identical_line_pairs_mask = torch.tensor([
        edit_data['stratum'] == Q4Stratum.IDENTICAL
        for edit_data in pair_edit_data
    ])
    ending_changed_mask = torch.tensor([
        edit_data['ending_changed']
        for edit_data in pair_edit_data
    ])
    
    q1_results = []
    primary_projections = {}
    for layer in experiment_config.target_layers:
        vectors = (
            tuple(DifferenceVector)
            if layer == experiment_config.primary_layer
            else experiment_config.q1_layer_sweep_vectors
        )
        for pooling_type in PoolingType:
            q1_inputs_dict = _load_q1_inputs_dict(
                pair, split_suffix, pooling_type, layer, vectors, experiment_config, run_dirs, processed_data_dir,
            )
            for vector, q1_inputs in q1_inputs_dict.items():
                projections = stats_utils.compute_fold_projections(
                    q1_inputs.minuend_activation_matrix - q1_inputs.subtrahend_activation_matrix,
                    q1_inputs.fold_ids,
                    q1_inputs.mean_vector_folds,
                )
                q1_results.append(_compute_q1_results(
                    projections, q1_inputs, identical_line_pairs_mask, ending_changed_mask, pair, vector, layer, pooling_type, experiment_config,
                ))
                if layer == experiment_config.primary_layer and vector == DifferenceVector.OFFSET:
                    primary_projections[pooling_type] = projections
    
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
    
    for pooling_type, projections in primary_projections.items():
        sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
        torch.save(projections, run_dirs.pair_results / f'{Q1_PROJECTIONS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}.pt')
    save_json(q1_results, run_dirs.pair_results, f'{Q1_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}')
    return primary_projections


### Q3 ###
def _get_edit_count_bin_ids(
    edit_counts: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    is_zero = edit_counts == 0
    if is_zero.double().mean() < 1 / num_bins:
        bin_edges = torch.quantile(
            edit_counts,
            torch.linspace(0, 1, num_bins + 1, dtype=edit_counts.dtype)[1:-1],
        )
        bin_ids = torch.searchsorted(bin_edges, edit_counts)
        return bin_ids
    
    nonzero_bin_edges = torch.quantile(
        edit_counts[~is_zero],
        torch.linspace(0, 1, num_bins, dtype=edit_counts.dtype)[1:-1],
    )
    bin_ids = torch.where(
        is_zero, 0, torch.searchsorted(nonzero_bin_edges, edit_counts) + 1,
    )
    return bin_ids


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
    bin_ids = _get_edit_count_bin_ids(edit_counts, experiment_config.q3_num_dose_bins)
    
    spearman_ci = bootstrap_fn(values, stats_utils.compute_spearman)
    spearman_p_value = float(spearmanr(projections, edit_counts).pvalue)
    partial_spearman_ci = bootstrap_fn(values, stats_utils.compute_partial_spearman)
    projection_mean_ci_by_bin = {
        bin_id: bootstrap_fn(projections[bin_ids == bin_id])
        for bin_id in bin_ids.unique().tolist()
    }
    count_results = Q3CountResults(
        spearman_ci=spearman_ci,
        spearman_p_value=spearman_p_value,
        spearman_p_value_fdr=None,
        partial_spearman_ci=partial_spearman_ci,
        projection_mean_ci_by_bin=projection_mean_ci_by_bin,
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
    for pooling_type in experiment_config.q3_q4_pooling_types:
        pooling_annotations_list = _get_pooling_annotations_list(pair_annotations_list, pooling_type)
        q3_results.append(_compute_q3_results(
            pair, pooling_type, primary_projections[pooling_type], pooling_annotations_list, experiment_config,
        ))

    count_results_list = [
        count_results
        for results in q3_results
        for count_results in results['count_results'].values()
    ]
    p_values_fdr = false_discovery_control([
        count_results['spearman_p_value']
        for count_results in count_results_list
    ])

    for count_results, p_value in zip(count_results_list, p_values_fdr):
        count_results['spearman_p_value_fdr'] = float(p_value)
    save_json(q3_results, out_dir, f'{Q3_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}')


### Q4 ###
def _compute_q4_parameter_cis(
    fit: edit_regression.Fit,
    regression_inputs: edit_regression.Inputs,
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4ParameterCIs:
    parameters = edit_regression.unpack_parameters(fit, regression_inputs.controls.shape[1])
    regression_shares = edit_regression.compute_shares(fit, regression_inputs)
    draws = edit_regression.bootstrap_regression(
        fit, regression_inputs, experiment_config.q4_num_bootstrap_samples, experiment_config.random_seed + SeedOffset.Q3_Q4_BOOTSTRAP,
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
    parameter_cis = Q4ParameterCIs(
        weights=weights,
        distance_decay_exponent=distance_decay_exponent,
        size_exponent=size_exponent,
        shares=shares,
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
    )
    return parameter_estimates


def _compute_q4_curve_results(
    regression_inputs: edit_regression.Inputs,
    curve: Q4CurveFunction,
    bootstrap: bool,
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
        _compute_q4_parameter_cis(fit, regression_inputs, experiment_config)
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
    )
    return curve_results


def _get_q4_verdict(
    weight_cis: list[CIStats],
) -> Q4EditVerdict:
    if all(weight_ci['ci_lower'] > 1 for weight_ci in weight_cis):
        return Q4EditVerdict.STRONGER
    if all(weight_ci['ci_upper'] < 1 for weight_ci in weight_cis):
        return Q4EditVerdict.WEAKER
    return Q4EditVerdict.NOT_DIFFERENT


def _get_q4_verdicts(
    curve_results: dict[Q4CurveFunction, Q4CurveResults],
    primary_curves: tuple[Q4CurveFunction, ...],
) -> dict[Q4EditKind, Q4EditVerdict]:
    verdicts = {
        kind: _get_q4_verdict([
            curve_results[curve]['weights'][kind]
            for curve in primary_curves
        ])
        for kind in Q4EditKind
        if kind != Q4EditKind.CONTENT_CHANGE
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


def _compute_q4_results(
    pair: SupervisedPair,
    pooling_type: PoolingType,
    projections: torch.Tensor,
    pooling_annotations_list: list[TextPairAnnotations],
    experiment_config: ExperimentConfig,
    edit_kinds: tuple[Q4EditKind, ...] = edit_regression.EDIT_KINDS,
) -> Q4Results:
    valued_pairs_mask = ~projections.isnan()
    pooling_annotations_list = [
        annotations
        for annotations, is_valued
        in zip(pooling_annotations_list, valued_pairs_mask.tolist())
        if is_valued
    ]
    pair_edit_data_list = [
        annotations['edit_data_by_threshold']['strict']
        for annotations in pooling_annotations_list
    ]
    regression_inputs = edit_regression.build_regression_inputs(
        projections[valued_pairs_mask],
        [annotations['edits'] for annotations in pooling_annotations_list],
        pair_edit_data_list,
        [annotations['controls'] for annotations in pooling_annotations_list],
        experiment_config.q4_format_flags,
        uses_distance_decay=(pooling_type == PoolingType.BOUNDARY),
    )
    is_bootstrap = pooling_type in experiment_config.q3_q4_pooling_types
    num_pairs_by_stratum, mean_projection_by_stratum = _compute_mean_projection_by_stratum(
        regression_inputs.projections, pair_edit_data_list, experiment_config,
    )
    num_edits_by_kind = {
        kind: int((regression_inputs.edit_kind_indices == kind_idx).sum())
        for kind_idx, kind in enumerate(edit_kinds)
    }
    curve_results = {
        curve:_compute_q4_curve_results(
            regression_inputs,
            curve,
            is_bootstrap and curve in experiment_config.q4_primary_curves,
            experiment_config,
        )
        for curve in Q4CurveFunction
    }
    verdicts = (
        _get_q4_verdicts(curve_results, experiment_config.q4_primary_curves)
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


### Main ###
def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    split_suffix: SplitSuffixes,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    for pair in supervised_pairs:
        pair_annotations_list = _load_pair_annotations_list(pair, split_suffix, experiment_config, run_dirs)
        pair_edit_data = _save_pair_edit_data(pair, split_suffix, pair_annotations_list, run_dirs.pair_results)
        primary_projections = _compute_and_save_q1_results(pair, split_suffix, pair_edit_data, experiment_config, run_dirs, processed_data_dir)
        _compute_and_save_q3_results(pair, split_suffix, primary_projections, pair_annotations_list, experiment_config, run_dirs.pair_results)
        _compute_and_save_q4_results(pair, split_suffix, primary_projections, pair_annotations_list, experiment_config, run_dirs.pair_results)


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    verify_extraction_log(EXPERIMENT_CONFIG, run_dirs)
    compute_and_save_metrics(CORPUS_CONFIG, SplitSuffixes.SUPERVISED, EXPERIMENT_CONFIG, run_dirs)
    tqdm.write('06: Pair Metrics Computation Complete\n')
