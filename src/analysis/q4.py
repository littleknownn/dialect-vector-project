from pathlib import Path
from typing import NamedTuple

from scipy.stats import false_discovery_control
import torch

from src.analysis.pair_data import get_pooling_annotations_list
from src.core.config_utils import (
    ExperimentConfig,
    SupervisedPair,
)
from src.core.paths import (
    Q4_RESULTS_PREFIX,
    get_sweep_identifier,
)
import src.core.edit_regression as edit_regression
import src.core.stats_utils as stats_utils
from src.core.utils import (
    CIStats,
    PairAnnotations,
    PairEditData,
    PoolingType,
    Q4CurveFunction,
    Q4CurveResults,
    Q4Edit,
    Q4EditKind,
    Q4EditVerdict,
    Q4Results,
    Q4Stratum,
    SeedOffset,
    SplitSuffixes,
    TextPairAnnotations,
    save_json,
)


class Q4ParameterCIs(NamedTuple):
    weights: dict[Q4EditKind, CIStats]
    distance_decay_exponent: CIStats
    size_exponent: CIStats
    shares: dict[Q4EditKind, CIStats]
    weight_p_values: dict[Q4EditKind, float] | None
    num_failed_fits: int


def compute_q4_parameter_cis(
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


def get_q4_parameter_estimates(
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
    fit = edit_regression.fit_regression(regression_inputs, curve, experiment_config.q4.regression_settings)
    predictions = edit_regression.predict(fit, regression_inputs).projections
    held_out_predictions = edit_regression.compute_held_out_predictions(
        fit, regression_inputs, experiment_config.q4.num_held_out_folds, experiment_config.random_seed + SeedOffset.Q4_HELD_OUT_FOLDS,
    )
    regression_shares = edit_regression.compute_shares(fit, regression_inputs)
    parameter_cis = (
        compute_q4_parameter_cis(fit, regression_inputs, experiment_config.q4.num_bootstrap_samples, experiment_config)
        if bootstrap
        else get_q4_parameter_estimates(fit, regression_inputs)
    )
    distances = (
        experiment_config.q4.marginal_effect_distances
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
        predictions, regression_inputs.projections, experiment_config.q4.num_calibration_bins,
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
    primary_curve_results = [curve_results[curve] for curve in experiment_config.q4.primary_curves]
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
            regression_inputs, curve, experiment_config.q4.regression_settings,
        ),
        regression_inputs,
        experiment_config.q4.num_held_out_folds,
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


def build_q4_regression_inputs(
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
        format_flags=experiment_config.q4.format_flags,
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
    regression_inputs = build_q4_regression_inputs(
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
    tested_edit_kinds = _get_q4_tested_edit_kinds(num_edits_by_kind, experiment_config.q4.min_edits_for_verdict)
    curve_results = {
        curve:_compute_q4_curve_results(
            regression_inputs,
            curve,
            is_bootstrap and curve in experiment_config.q4.primary_curves,
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
            experiment_config.q4.num_bootstrap_samples,
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


def compute_and_save_q4_results(
    pair: SupervisedPair,
    split_suffix: SplitSuffixes,
    primary_projections: dict[PoolingType, torch.Tensor],
    pair_annotations_list: list[PairAnnotations],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    for pooling_type in experiment_config.q3_q4_pooling_types:
        sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
        pooling_annotations_list = get_pooling_annotations_list(pair_annotations_list, pooling_type)
        q4_results = _compute_q4_results(pair, pooling_type, primary_projections[pooling_type], pooling_annotations_list, experiment_config)
        save_json(q4_results, out_dir, f'{Q4_RESULTS_PREFIX}{pair.corpus_name}{split_suffix}_{sweep_identifier}')
