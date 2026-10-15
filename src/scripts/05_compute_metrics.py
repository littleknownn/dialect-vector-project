from collections import defaultdict
from collections.abc import Callable
import itertools
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm
from scipy.stats import false_discovery_control
import torch

from src.core.config import (
    DatasetConfig,
    ExperimentConfig,
    SupervisedPair,
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
    get_estimator_data,
)
import src.core.math_utils as math_utils
from src.core.paths import (
    ACTIVATIONS_DIR,
    BEHAVIORAL_RESULTS_FILENAME,
    ESTIMATORS_DIR,
    H1_FLOORS_PREFIX,
    H1_NULL_PREFIX,
    H1_RESULTS_FILENAME,
    H1_SVD_PREFIX,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    Q1_FOLD_IDS_PREFIX,
    Q1_FOLD_MEAN_VECTORS_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    PAIR_EDIT_DATA_PREFIX,
    SWEEPS_DIR,
    SWEEP_RESULTS_FILENAME,
    get_sweep_identifier,
)
import src.core.stats_utils as stats_utils
import src.core.text_utils as text_utils
from src.core.utils import (
    AucSummary,
    BehavioralSummary,
    CIStats,
    FdrSummary,
    H1Results,
    MetricTypeEnum,
    PairEditData,
    PoolingTypeEnum,
    Q1ConsistencyLabel,
    Q1LabeledThresholds,
    Q1Results,
    Q4GroupLabelThresholds,
    Q4Results,
    Q4Stratum,
    SingleEstimatorPrefixes,
    SweepKey,
    SweepSummary,
    load_json,
    load_jsonl,
    save_json,
)


class Q4FeatureMatrix(NamedTuple):
    matrix: torch.Tensor
    group_feature_indices: dict[str, list[int]]

class SteeringCurvePoint(NamedTuple):
    alpha: float
    value: float
    p_value: float | None


class BehavioralResults(NamedTuple):
    auc_pos: dict[SweepKey, float]
    auc_neg: dict[SweepKey, float]
    p_values_fdr: dict[SweepKey, dict[float, float]]


class H1Data(NamedTuple):
    svd_tensors: dict[str, torch.Tensor]
    nulls: torch.Tensor
    floors: dict[str, dict[str, list[float]]]

### Pair Edit Data ###
def _compute_and_save_pair_edit_data(
    pair: SupervisedPair,
    experiment_config: ExperimentConfig,
    processed_data_dir: Path,
    out_dir: Path,
) -> list[PairEditData]:
    aae_lines = load_jsonl(processed_data_dir / f'{pair.aae_split}.jsonl')
    sae_lines = load_jsonl(processed_data_dir / f'{pair.sae_split}.jsonl')
    pair_edit_data = [
        text_utils.compute_pair_edit_data(aae_line, sae_line, experiment_config.q4_orthographic_threshold, experiment_config.q4_function_words)
        for aae_line, sae_line in zip(aae_lines, sae_lines)
    ]
    save_json(pair_edit_data, out_dir, f'{PAIR_EDIT_DATA_PREFIX}{pair.corpus_name}')
    return pair_edit_data


### Q1 ###
def _compute_q1_fold_projections(
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    experiment_config: ExperimentConfig,
    activations_dir: Path,
    estimators_dir: Path,
) -> torch.Tensor:
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
    aae_activation_matrix = torch.load(activations_dir / f'{pair.aae_split}_{sweep_identifier}.pt', weights_only=True).float()
    sae_activation_matrix = torch.load(activations_dir / f'{pair.sae_split}_{sweep_identifier}.pt', weights_only=True).float()
    
    fold_ids = torch.load(estimators_dir / f'{Q1_FOLD_IDS_PREFIX}{pair.corpus_name}.pt', weights_only=True)
    mean_vector_folds = torch.load(estimators_dir / f'{Q1_FOLD_MEAN_VECTORS_PREFIX}{pair.corpus_name}_{sweep_identifier}.pt', weights_only=True, map_location='cpu')
    fold_projections = stats_utils.compute_fold_projections(aae_activation_matrix - sae_activation_matrix, fold_ids, mean_vector_folds)
    return fold_projections


def _compute_q1_results(
    projections: torch.Tensor,
    identical_pairs_mask: torch.Tensor,
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    experiment_config: ExperimentConfig,
) -> Q1Results:
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

    mean_projection = stats_utils.compute_bootstrap_ci(
        projections,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + 1, #
    )
    positive_projection_ratio = stats_utils.compute_bootstrap_ci(
        projections,
        stats_utils.compute_positive_projection_ratio,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + 1, #
    )
    standardized_projection_mean = stats_utils.compute_bootstrap_ci(
        projections,
        stats_utils.compute_standardized_projection_mean,
        num_draws=experiment_config.num_bootstrap_samples,
        p_alpha=experiment_config.p_alpha,
        seed=experiment_config.random_seed + 1, #
    )
    label = _get_q1_label(
        positive_projection_ratio['estimate'],
        standardized_projection_mean['ci_lower'],
        experiment_config.q1_labeled_bounds,
    )
    identical_pairs_median_abs_projection = float(projections[identical_pairs_mask].abs().median())
    other_pairs_median_abs_projection = float(projections[~identical_pairs_mask].abs().median())

    results = Q1Results(
        corpus_name=pair.corpus_name,
        pooling_type=pooling_type,
        num_pairs=len(projections),
        mean_projection=mean_projection,
        positive_projection_ratio=positive_projection_ratio,
        standardized_projection_mean=standardized_projection_mean,
        label=label,
        identical_pairs_median_abs_projection=identical_pairs_median_abs_projection,
        other_pairs_median_abs_projection=other_pairs_median_abs_projection,
    )
    return results


def _compute_and_save_q1_results(
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    projections: torch.Tensor,
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
    identical_line_pairs_mask = torch.tensor([
        edit_data['stratum'] == Q4Stratum.IDENTICAL for edit_data in pair_edit_data
    ])
    q1_results = _compute_q1_results(projections, identical_line_pairs_mask, pair, pooling_type, experiment_config)
    
    torch.save(projections, out_dir / f'{Q1_PROJECTIONS_PREFIX}{pair.corpus_name}_{sweep_identifier}.pt')
    save_json(q1_results, out_dir, f'{Q1_RESULTS_PREFIX}{pair.corpus_name}_{sweep_identifier}')    


### Q4 ###
def _build_q4_feature_matrix(
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
) -> Q4FeatureMatrix:
    feature_vectors = [torch.ones(len(pair_edit_data), dtype=torch.float64)]
    group_feature_indices = {'intercept': [0]}
    for group_name, feature_names in experiment_config.q4_tags_per_group._asdict().items():
        group_feature_indices[group_name] = []
        for feature_name in feature_names:
            feature_vector = torch.tensor(
                [float(edit_data[feature_name]) for edit_data in pair_edit_data],
                dtype=torch.float64,
            )
            if feature_name in experiment_config.q4_log_centered_features:
                log_feature_vector = feature_vector.log()
                feature_vector = log_feature_vector - log_feature_vector.mean()
            group_feature_indices[group_name].append(len(feature_vectors))
            feature_vectors.append(feature_vector)
    feature_matrix = Q4FeatureMatrix(
        matrix=torch.stack(feature_vectors, dim=1),
        group_feature_indices=group_feature_indices,
    )
    return feature_matrix


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
        mean_projection_by_stratum[stratum] = stats_utils.compute_bootstrap_ci(
            values=projections[stratum_mask],
            num_draws=experiment_config.num_bootstrap_samples,
            p_alpha=experiment_config.p_alpha,
            seed=experiment_config.random_seed + 1, #
        )
    return num_pairs_by_stratum, mean_projection_by_stratum


def _compute_attribution_ci(
    feature_matrix: torch.Tensor,
    projections_float64: torch.Tensor,
    group_names: list[str],
    group_feature_indices: list[list[int]],
    group_statistic: stats_utils.GroupStatistic,
    experiment_config: ExperimentConfig,
) -> dict[str, CIStats]:
    regression_data = torch.cat([feature_matrix, projections_float64.unsqueeze(1)], dim=1)
    estimates = group_statistic(feature_matrix, projections_float64, group_feature_indices)
    draws = stats_utils.compute_bootstrap_draws(
        regression_data,
        lambda data: group_statistic(data[..., :-1], data[..., -1], group_feature_indices),
        experiment_config.num_bootstrap_samples,
        experiment_config.random_seed + 1,
    )
    attribution_ci = {
        group_name: stats_utils.compute_percentile_ci(
            draws[:, group_idx],
            experiment_config.p_alpha,
            float(estimates[group_idx])
        )
        for group_idx, group_name in enumerate(group_names)
    }
    return attribution_ci


def _compute_q4_results(
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    projections: torch.Tensor,
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
) -> Q4Results:
    def _get_q4_labels(
        group_shares: dict[str, CIStats],
        unique_r_squared: dict[str, CIStats],
        thresholds: Q4GroupLabelThresholds,
    ) -> tuple[list[str], list[str]]:
        contributing_groups = [
            group_name for group_name, r_squared_ci in unique_r_squared.items()
            if group_shares[group_name]['ci_lower'] > thresholds.contribution_share_ci_lower
            and r_squared_ci['ci_lower'] > thresholds.contribution_unique_r_squared_ci_lower
        ]
        dominating_groups = [
            group_name for group_name in contributing_groups
            if group_shares[group_name]['estimate'] >= thresholds.dominant_share
        ]
        return contributing_groups, dominating_groups
    
    projections_float64 = projections.double()
    feature_matrix = _build_q4_feature_matrix(pair_edit_data, experiment_config)
    all_groups = list(feature_matrix.group_feature_indices)
    feature_groups = [group_name for group_name in all_groups if group_name != 'intercept']

    num_pairs_by_stratum, mean_projection_by_stratum = _compute_mean_projection_by_stratum(projections, pair_edit_data, experiment_config)
    group_shares = _compute_attribution_ci(
        feature_matrix.matrix,
        projections_float64,
        all_groups,
        [feature_matrix.group_feature_indices[group_name] for group_name in all_groups],
        stats_utils.compute_group_shares,
        experiment_config,
    )
    unique_r_squared = _compute_attribution_ci(
        feature_matrix.matrix,
        projections_float64,
        feature_groups,
        [feature_matrix.group_feature_indices[group_name] for group_name in feature_groups],
        stats_utils.compute_unique_r_squared,
        experiment_config,
    )
    r_squared = float(stats_utils.compute_r_squared(feature_matrix.matrix, projections_float64))
    contributing_groups, dominating_groups = _get_q4_labels(group_shares, unique_r_squared, experiment_config.q4_group_label_thresholds)

    results = Q4Results(
        corpus_name=pair.corpus_name,
        pooling_type=pooling_type,
        num_pairs_by_stratum=num_pairs_by_stratum,
        mean_projection_by_stratum=mean_projection_by_stratum,

        group_shares=group_shares,
        r_squared=r_squared,
        unique_r_squared=unique_r_squared,
        contributing_groups=contributing_groups,
        dominating_groups=dominating_groups,
    )
    return results



def _compute_and_save_q4_results(
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    projections: torch.Tensor,
    pair_edit_data: list[PairEditData],
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
    q4_results = _compute_q4_results(pair, pooling_type, projections, pair_edit_data, experiment_config)
    save_json(q4_results, out_dir, f'{Q4_RESULTS_PREFIX}{pair.corpus_name}_{sweep_identifier}')

### Steering Metrics ###
def _load_steering_metric_sweeps(
    in_dir: Path
) -> dict[SweepKey, list[SteeringCurvePoint]]:
    steering_metric_sweeps = defaultdict(list)
    for filepath in in_dir.glob('*.json'):
        metrics = load_json(filepath)
        for metric_data in metrics: # deserialization boundary
            metric_data['pooling_type'] = PoolingTypeEnum(metric_data['pooling_type'])
            metric_data['metric'] = MetricTypeEnum(metric_data['metric'])

            sweep_key = SweepKey(
                **{key: metric_data[key] for key in SweepKey._fields}
            )
            point = SteeringCurvePoint(
                **{key: metric_data.get(key) for key in SteeringCurvePoint._fields}
            )
            steering_metric_sweeps[sweep_key].append(point)

    for points in steering_metric_sweeps.values():
        points.sort(key=lambda point: point.alpha)

    return steering_metric_sweeps


def _flatten_sweep_metrics(
    steering_metric_sweeps: dict[SweepKey, list[SteeringCurvePoint]],
) -> list[SweepSummary]:
    summary = []
    for sweep_key, points in steering_metric_sweeps.items():
        for point in points:
            summary.append(SweepSummary(
                layer=sweep_key.layer,
                pooling_type=sweep_key.pooling_type,
                rank=sweep_key.rank,
                estimator_name=sweep_key.estimator_name,
                corpus_name=sweep_key.corpus_name,
                metric=sweep_key.metric,
                alpha=point.alpha,
                value=point.value,
                p_value=point.p_value,
            ))
    return summary


def _compute_behavioral_results(
    steering_metric_sweeps: dict[SweepKey, list[SteeringCurvePoint]],
) -> BehavioralResults:
    auc_pos_results = {}
    auc_neg_results = {}

    llr_keys = []
    llr_p_values = []
    for sweep_key, curve_points in tqdm(steering_metric_sweeps.items(), desc='Computing Behavioral Results'):
        pos_points = [p for p in curve_points if p.alpha >= 0.0]
        neg_points = [p for p in curve_points if p.alpha <= 0.0]

        auc_pos_results[sweep_key] = float(np.trapz(
            y=[point.value for point in pos_points],
            x=[point.alpha for point in pos_points],
        ))
        auc_neg_results[sweep_key] = -float(np.trapz(
            y=[point.value for point in neg_points],
            x=[point.alpha for point in neg_points],
        ))

        if sweep_key.metric == MetricTypeEnum.LLR_SHIFT:
            for point in curve_points:
                if point.p_value is not None:
                    llr_p_values.append(point.p_value)
                    llr_keys.append((sweep_key, point.alpha))

    fdr_results = defaultdict(dict)
    if llr_p_values:
        llr_p_values_fdr = false_discovery_control(llr_p_values)
        for (sweep_key, alpha), p_value_fdr in zip(llr_keys, llr_p_values_fdr):
            fdr_results[sweep_key][alpha] = float(p_value_fdr)

    results = BehavioralResults(
        auc_pos=auc_pos_results,
        auc_neg=auc_neg_results,
        p_values_fdr=dict(fdr_results),
    )
    return results


def _flatten_behavioral_results(behavioral_results: BehavioralResults) -> BehavioralSummary:
    auc = []
    for sweep_key, auc_pos in behavioral_results.auc_pos.items():
        auc.append(AucSummary(
            layer=sweep_key.layer,
            pooling_type=sweep_key.pooling_type,
            rank=sweep_key.rank,
            estimator_name=sweep_key.estimator_name,
            corpus_name=sweep_key.corpus_name,
            metric=sweep_key.metric,
            auc_pos=auc_pos,
            auc_neg=behavioral_results.auc_neg[sweep_key],
        ))

    fdr = []
    for sweep_key, alpha_dict in behavioral_results.p_values_fdr.items():
        for alpha, p_value_fdr in alpha_dict.items():
            fdr.append(FdrSummary(
                layer=sweep_key.layer,
                pooling_type=sweep_key.pooling_type,
                rank=sweep_key.rank,
                estimator_name=sweep_key.estimator_name,
                corpus_name=sweep_key.corpus_name,
                metric=sweep_key.metric,
                alpha=alpha,
                p_value_fdr=p_value_fdr,
            ))

    summary = BehavioralSummary(auc=auc, fdr=fdr)
    return summary


def _compute_and_save_steering_metrics(
    sweep_results_filename: str,
    behavioral_results_filename: str,
    in_dir: Path,
    out_dir: Path,
) -> None:
    steering_metric_sweeps = _load_steering_metric_sweeps(in_dir)
    behavioral_results = _compute_behavioral_results(steering_metric_sweeps)
    save_json(_flatten_sweep_metrics(steering_metric_sweeps), out_dir, sweep_results_filename)
    save_json(_flatten_behavioral_results(behavioral_results), out_dir, behavioral_results_filename)


### H1 ###
def _load_h1_data(
    pooling_type: PoolingTypeEnum,
    experiment_config: ExperimentConfig,
    in_dir: Path,
) -> H1Data:
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)

    svd_tensors = {
        corpus_name: torch.load(
            in_dir / f'{H1_SVD_PREFIX}{corpus_name}_{sweep_identifier}.pt',
            weights_only=True,
        )
        for corpus_name in experiment_config.h1_corpora
    }
    null_tensors = torch.load(in_dir / f'{H1_NULL_PREFIX}{sweep_identifier}.pt', weights_only=True)
    
    floors_filepath = in_dir / f'{H1_FLOORS_PREFIX}{sweep_identifier}.json'
    floors = load_json(floors_filepath)

    h1_data = H1Data(
        svd_tensors=svd_tensors,
        nulls=null_tensors,
        floors=floors,
    )
    return h1_data

def _compute_h1_results(
    h1_data: H1Data,
    pooling_type: PoolingTypeEnum,
    experiment_config:  ExperimentConfig,
    in_dir: Path,
    supervised_corpus_name: str,
    estimator_rank: int,
) -> H1Results:
    act_add_sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type, 1)
    supervised_act_add_vector = torch.load(
        in_dir / f'{SingleEstimatorPrefixes.ACT_ADD}{supervised_corpus_name}_{act_add_sweep_identifier}.pt',
        weights_only=True,
    )

    floors = {
        corpus_name: h1_data.floors[corpus_name][str(estimator_rank)]
        for corpus_name in experiment_config.h1_corpora
    }

    # fix hardcode
    pair_distances = {
        f'{corpus_a_name}|{corpus_b_name}': math_utils.compute_geometric_metrics(
            h1_data.svd_tensors[corpus_a_name][:, :, :estimator_rank],
            h1_data.svd_tensors[corpus_b_name][:, :, :estimator_rank],
        ).chordal_distance.tolist()
        for corpus_a_name, corpus_b_name in itertools.combinations(experiment_config.h1_corpora, 2)
    }

    num_null_pairs = len(h1_data.nulls) // 2
    null_distances = math_utils.compute_geometric_metrics(
        h1_data.nulls[:num_null_pairs, :, :estimator_rank],
        h1_data.nulls[num_null_pairs:2 * num_null_pairs, :, :estimator_rank],
    ).chordal_distance.tolist()

    projection_energies = {
        corpus_name: math_utils.compute_vector_energy(
            svd_tensor[:, :, :estimator_rank],
            supervised_act_add_vector,
        ).tolist()
        for corpus_name, svd_tensor in h1_data.svd_tensors.items()
    }
    projection_energies['null'] = math_utils.compute_vector_energy(
        h1_data.nulls[:, :, :estimator_rank],
        supervised_act_add_vector,
    ).tolist()
    
    results = H1Results(
        supervised_vector_norm=float(torch.linalg.vector_norm(supervised_act_add_vector)),
        floors=floors,
        pair_distances=pair_distances,
        null_distances=null_distances,
        projection_energies=projection_energies,
    )
    return results


def _compute_and_save_h1_results(
    pair: SupervisedPair,
    pooling_type: PoolingTypeEnum,
    experiment_config: ExperimentConfig,
    h1_results_filename: str,
    estimators_dir: Path,
    out_dir: Path,
) -> None:
    max_rank = max(experiment_config.target_ranks)
    h1_data = _load_h1_data(pooling_type, experiment_config, estimators_dir)
    h1_results = _compute_h1_results(h1_data, pooling_type, experiment_config, estimators_dir, pair.corpus_name, max_rank)
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type, max_rank)
    save_json(h1_results, out_dir, f'{h1_results_filename}_{pair.corpus_name}_{sweep_identifier}')


### Main ###
def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    h1_results_filename: str = H1_RESULTS_FILENAME,
    sweep_results_filename: str = SWEEP_RESULTS_FILENAME,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    activations_dir: Path = ACTIVATIONS_DIR,
    behavioral_results_filename: str = BEHAVIORAL_RESULTS_FILENAME,
    estimators_dir: Path = ESTIMATORS_DIR,
    sweeps_dir: Path = SWEEPS_DIR,
    out_dir: Path = METRICS_DIR,
) -> None:
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    for pair in supervised_pairs:
        pair_edit_data = _compute_and_save_pair_edit_data(pair, experiment_config, processed_data_dir, out_dir)
        for pooling_type in PoolingTypeEnum:
            q1_projections = _compute_q1_fold_projections(pair, pooling_type, experiment_config, activations_dir, estimators_dir)
            _compute_and_save_q1_results(pair, pooling_type, q1_projections, pair_edit_data, experiment_config, out_dir)
            _compute_and_save_q4_results(pair, pooling_type, q1_projections, pair_edit_data, experiment_config, out_dir)
            _compute_and_save_h1_results(pair, pooling_type, experiment_config, h1_results_filename, estimators_dir, out_dir)

    _compute_and_save_steering_metrics(sweep_results_filename, behavioral_results_filename, sweeps_dir, out_dir)


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    compute_and_save_metrics(CORPUS_CONFIG, EXPERIMENT_CONFIG)
    tqdm.write('05: Metric Computation Complete\n')
