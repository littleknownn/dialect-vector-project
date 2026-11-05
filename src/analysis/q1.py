from pathlib import Path
from typing import NamedTuple

from scipy.stats import false_discovery_control
import torch
import torch.nn.functional as F

from src.analysis.pair_data import get_identical_pairs_mask
from src.core.config_utils import (
    ExperimentConfig,
    Q1LabeledThresholds,
    SupervisedPair,
)
from src.core.paths import (
    FOLD_IDS_PREFIX,
    Q1_PROJECTIONS_PREFIX,
    Q1_SHUFFLED_PROJECTIONS_PREFIX,
    Q1_RESULTS_PREFIX,
    RunDirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
import src.core.stats_utils as stats_utils
from src.core.utils import (
    DifferenceVector,
    PairAnnotations,
    PairEditData,
    PoolingType,
    Q1ConsistencyLabel,
    Q1Results,
    SeedOffset,
    SplitSuffixes,
    load_json,
    save_json,
)



class Q1Inputs(NamedTuple):
    minuend_activation_matrix: torch.Tensor
    subtrahend_activation_matrix: torch.Tensor
    fold_ids: torch.Tensor
    mean_vector_folds: torch.Tensor


class Q1Projections(NamedTuple):
    dialect_projections_by_pooling: dict[PoolingType, torch.Tensor]
    primary_pooling_projections_by_vector: dict[DifferenceVector, torch.Tensor]


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
        for _ in range(experiment_config.q1.num_shuffled_pairings)
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
        experiment_config.q1.labeled_bounds,
    )

    mean_squared_norm = torch.linalg.vector_norm(differences, dim=1).square().mean()
    activation_difference_energy = float(torch.linalg.vector_norm(differences.mean(dim=0)).square() / mean_squared_norm)
    projection_energy = float(projections.double().square().mean() / mean_squared_norm)
    content_matching_gain = float(
        1 - projections.double().var() / shuffled_projections.var(dim=1).mean()
    )

    sign_flip_p_value = stats_utils.compute_sign_flip_p_value(projections, experiment_config.q1.num_sign_flips, experiment_config.random_seed + SeedOffset.SIGN_FLIP)
    
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


def compute_and_save_q1_results(
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
            else experiment_config.q1.layer_sweep_vectors
        )
        for pooling_type in PoolingType:
            identical_pairs_mask = get_identical_pairs_mask(pair_annotations_list, pooling_type)
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
        and results['vector'] in experiment_config.q1.fdr_vectors
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