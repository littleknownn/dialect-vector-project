from pathlib import Path
from typing import NamedTuple

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
    Q4GroupLabelThresholds,
    SupervisedPair,
    get_estimator_data,
)
from src.core.paths import (
    PROCESSED_DATA_DIR,
    Q1_FOLD_IDS_PREFIX,
    Q1_FOLD_MEAN_VECTORS_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    Q4_RESULTS_PREFIX,
    PAIR_EDIT_DATA_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
    verify_extraction_log,
)
import src.core.stats_utils as stats_utils
import src.core.text_utils as text_utils
from src.core.utils import (
    CIStats,
    PairEditData,
    PoolingTypeEnum,
    Q1ConsistencyLabel,
    Q1Results,
    Q4Results,
    Q4Stratum,
    load_jsonl,
    save_json,
)


class Q4FeatureMatrix(NamedTuple):
    matrix: torch.Tensor
    group_feature_indices: dict[str, list[int]]


### Pair Edit Data ###
def _compute_and_save_pair_edit_data(
    pair: SupervisedPair,
    experiment_config: ExperimentConfig,
    processed_data_dir: Path,
    out_dir: Path,
) -> list[PairEditData]:
    aae_lines = load_jsonl(processed_data_dir / f'{pair.aae_split}.jsonl')[:experiment_config.debug_max_split_lines]
    sae_lines = load_jsonl(processed_data_dir / f'{pair.sae_split}.jsonl')[:experiment_config.debug_max_split_lines]
    pair_edit_data = [
        text_utils.compute_pair_edit_data(aae_line, sae_line, experiment_config.q4_orthographic_thresholds.strict, experiment_config.q4_function_words)
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
    q1_estimators_dir: Path,
) -> torch.Tensor:
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type)
    mean_vector_folds = torch.load(q1_estimators_dir / f'{Q1_FOLD_MEAN_VECTORS_PREFIX}{pair.corpus_name}_{sweep_identifier}.pt', weights_only=True, map_location='cpu')
    fold_ids = torch.load(q1_estimators_dir / f'{Q1_FOLD_IDS_PREFIX}{pair.corpus_name}.pt', weights_only=True)
    aae_activation_matrix = torch.load(
        get_sweep_tensor_filepath(activations_dir, sweep_identifier, pair.aae_split),
        weights_only=True,
    ).float()
    sae_activation_matrix = torch.load(
        get_sweep_tensor_filepath(activations_dir, sweep_identifier, pair.sae_split),
        weights_only=True,
    ).float()

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

    valued_pairs_mask = ~projections.isnan()
    num_excluded_pairs = len(projections) - int(valued_pairs_mask.sum())
    projections = projections[valued_pairs_mask]
    identical_pairs_mask = identical_pairs_mask[valued_pairs_mask]

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
        num_excluded_pairs=num_excluded_pairs,
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

    valued_pairs_mask = ~projections.isnan()
    pair_edit_data = [
        edit_data for edit_data, is_valued
        in zip(pair_edit_data, valued_pairs_mask.tolist())
        if is_valued
    ]
    projections = projections[valued_pairs_mask]
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


### Main ###
def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    for pair in supervised_pairs:
        pair_edit_data = _compute_and_save_pair_edit_data(pair, experiment_config, processed_data_dir, run_dirs.pair_results)
        for pooling_type in PoolingTypeEnum:
            q1_projections = _compute_q1_fold_projections(pair, pooling_type, experiment_config, run_dirs.activations, run_dirs.q1_estimators)
            _compute_and_save_q1_results(pair, pooling_type, q1_projections, pair_edit_data, experiment_config, run_dirs.pair_results)
            _compute_and_save_q4_results(pair, pooling_type, q1_projections, pair_edit_data, experiment_config, run_dirs.pair_results)


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    verify_extraction_log(EXPERIMENT_CONFIG, run_dirs)
    compute_and_save_metrics(CORPUS_CONFIG, EXPERIMENT_CONFIG, run_dirs)
    tqdm.write('06: Pair Metrics Computation Complete\n')
