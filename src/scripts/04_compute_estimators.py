import itertools
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm import tqdm

from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
)
from src.core.config_utils import (
    BaselineDatasetConfig,
    DatasetConfig,
    ExperimentConfig,
    SupervisedPair,
    get_estimator_data,
)
import src.core.math_utils as math_utils
from src.core.paths import (
    DIFFERENCE_VECTOR_PREFIX,
    PROCESSED_DATA_DIR,
    Q1_FOLD_IDS_PREFIX,
    Q1_FOLD_MEAN_VECTORS_PREFIX,
    H1_FLOORS_PREFIX,
    H1_NULL_PREFIX,
    H1_SVD_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
    verify_extraction_log,
)
import src.core.stats_utils as stats_utils
from src.core.utils import (
    DifferenceVector,
    PoolingTypeEnum,
    SplitSuffixes,
    load_json,
    save_json,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SplitLists(NamedTuple):
    supervised_pairs: list[SupervisedPair]
    supervised_split_names: list[str]
    aae_corpora_split_names: list[str]
    all_eval_split_names: list[str]


class PairMasks(NamedTuple):
    variants: dict[DifferenceVector, torch.Tensor]
    aae_back_valid: torch.Tensor


class Estimators(NamedTuple):
    difference_vectors: dict[str, torch.Tensor]
    fold_mean_vectors: dict[str, torch.Tensor]


def _load_activation_matrix(
    split_name: str,
    sweep_identifier: str,
    activations_dir: Path,
) -> torch.Tensor:
    activation_matrix = torch.load(
        get_sweep_tensor_filepath(activations_dir, sweep_identifier, split_name),
        weights_only=True,
    ).to(device, dtype=torch.float32)
    return activation_matrix


def _load_supervised_pair_masks(
    pair: SupervisedPair,
    experiment_config: ExperimentConfig,
    text_annotations_dir: Path,
    processed_data_dir: Path,
) -> PairMasks:
    pair_metadata = load_json(
        processed_data_dir / f'{pair.corpus_name}{SplitSuffixes.PAIR_METADATA}{SplitSuffixes.SUPERVISED}.json',
    )[:experiment_config.debug_max_split_lines]
    aae_annotations = load_json(text_annotations_dir / f'{pair.aae_split}.json')[:experiment_config.debug_max_split_lines]
    sae_annotations = load_json(text_annotations_dir / f'{pair.sae_split}.json')[:experiment_config.debug_max_split_lines]
    line_annotation_pairs = list(zip(aae_annotations, sae_annotations, strict=True))

    variant_masks = {
        DifferenceVector.NO_TYPOGRAPHY: torch.tensor([
            not metadata['is_sae_typography_normalized']
            for metadata in pair_metadata
        ]),
        DifferenceVector.SAME_TERMINAL_PUNCTUATION: torch.tensor([
            aae_line['has_terminal_punctuation'] == sae_line['has_terminal_punctuation']
            for aae_line, sae_line in line_annotation_pairs
        ]),
        DifferenceVector.NO_PROFANITY: torch.tensor([
            not any(
                line['profanity_matches'] or line['has_masked_word']
                for line in (aae_line, sae_line)
            )
            for aae_line, sae_line in line_annotation_pairs
        ]),
    }
    aae_back_valid = torch.tensor([
        metadata['is_aae_back_valid']
        for metadata in pair_metadata
    ])
    pair_masks = PairMasks(
        variants=variant_masks,
        aae_back_valid=aae_back_valid,
    )

    return pair_masks


def _get_split_lists(
    corpus_config: dict[str, DatasetConfig],
) -> SplitLists:
    supervised_pairs, aae_corpora_split_names = get_estimator_data(corpus_config)
    supervised_split_names = (
        [pair.aae_split for pair in supervised_pairs]
        + [pair.sae_split for pair in supervised_pairs]
    )
    baseline_split_names = [
        corpus_name for corpus_name, config in corpus_config.items()
        if isinstance(config, BaselineDatasetConfig)
    ]

    all_eval_split_names = (
        aae_corpora_split_names
        + supervised_split_names
        + baseline_split_names
    )

    split_lists = SplitLists(
        supervised_pairs=supervised_pairs,
        supervised_split_names=supervised_split_names,
        aae_corpora_split_names=aae_corpora_split_names,
        all_eval_split_names=all_eval_split_names,
    )
    return split_lists


def _save_estimators(
    estimators: dict[str, torch.Tensor],
    layer: int,
    pooling_type: PoolingTypeEnum,
    out_dir: Path,
) -> None:
    sweep_identifier = get_sweep_identifier(layer, pooling_type, 1)
    for estimator_name, estimator_matrix in estimators.items():
        torch.save(
            estimator_matrix,
            get_sweep_tensor_filepath(out_dir, sweep_identifier, estimator_name),
        )


def _compute_and_save_h1_tensors(
    activations: dict[str, torch.Tensor],
    sweep_identifier: str,
    max_rank: int,
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    tqdm.write(f'Computing H1 Calibrations for Sweep {sweep_identifier}...')
    torch.manual_seed(experiment_config.random_seed) #
    activation_matrices = {
        split_name: activations[split_name]
        for split_name in experiment_config.h1_corpora
    }

    min_matrix_len = min(len(matrix) for matrix in activation_matrices.values())
    min_half_len = min_matrix_len // 2

    svd_tensors = {
        corpus_name: []
        for corpus_name in experiment_config.h1_corpora
    }
    floors = {
        corpus_name: {
            rank: []
            for rank in range(1, max_rank + 1)
        }
        for corpus_name in experiment_config.h1_corpora
    }

    for _, (corpus_name, matrix) in itertools.product(
        range(experiment_config.num_equal_n_repeats),
        activation_matrices.items(),
    ):
        indices = torch.randperm(len(matrix), device=device)
        half_1, half_2 = matrix[indices[:min_half_len]], matrix[indices[min_half_len:min_matrix_len]] #

        basis_1 = math_utils.compute_svd_estimator(half_1, max_rank)
        basis_2 = math_utils.compute_svd_estimator(half_2, max_rank)

        for rank in range(1, max_rank + 1):
            rank_basis_1 = basis_1[:, :rank].contiguous()
            rank_basis_2 = basis_2[:, :rank].contiguous()
            chordal_distance = math_utils.compute_geometric_metrics(rank_basis_1, rank_basis_2).chordal_distance.item()
            floors[corpus_name][rank].append(chordal_distance)
        svd_tensors[corpus_name].append(basis_1)

    centered_pooled_activations_matrix = torch.cat([
        matrix[:min_half_len] - matrix[:min_half_len].mean(dim=0)
        for matrix in activation_matrices.values()
    ])

    null_tensors = [
        math_utils.compute_matched_covariance_matrix(centered_pooled_activations_matrix, max_rank)
        for _ in range(experiment_config.num_null_draws)
    ]
    torch.save(torch.stack(null_tensors), out_dir / f'{H1_NULL_PREFIX}{sweep_identifier}.pt')

    for corpus_name, tensors in svd_tensors.items():
        torch.save(torch.stack(tensors), out_dir / f'{H1_SVD_PREFIX}{corpus_name}_{sweep_identifier}.pt')

    save_json(floors, out_dir, f'{H1_FLOORS_PREFIX}{sweep_identifier}')


def _compute_difference_matrices(
    pair: SupervisedPair,
    activations:  dict[str, torch.Tensor],
    pair_masks: PairMasks,
    aae_back_activation_matrix: torch.Tensor | None,
) -> dict[DifferenceVector, torch.Tensor]:
    mask_rows_fn = lambda matrix, mask: torch.where(
        condition=mask.to(device).unsqueeze(1),
        input=matrix,
        other=float('nan'), #
    )
    difference_matrix = activations[pair.aae_split] - activations[pair.sae_split]
    difference_matrices = {
        DifferenceVector.OFFSET: difference_matrix
    }
    difference_matrices |= {
        vector: mask_rows_fn(difference_matrix, mask)
        for vector, mask in pair_masks.variants.items()
    }
    if aae_back_activation_matrix is not None:
        difference_matrices[DifferenceVector.AUTHORSHIP] = mask_rows_fn(
            activations[pair.aae_split] - aae_back_activation_matrix,
            pair_masks.aae_back_valid,
        )
        difference_matrices[DifferenceVector.DIALECT] = mask_rows_fn(
            aae_back_activation_matrix - activations[pair.sae_split],
            pair_masks.aae_back_valid,
        )
    return difference_matrices


def _save_q1_folds(
    fold_ids_dict: dict[str, torch.Tensor],
    mean_vector_folds_dict: dict[str, torch.Tensor],
    sweep_identifier: str,
    out_dir: Path,
) -> None:
    for corpus_name, fold_ids in fold_ids_dict.items():
        torch.save(fold_ids, out_dir / f'{Q1_FOLD_IDS_PREFIX}{corpus_name}.pt')
    for name, folds in mean_vector_folds_dict.items():
        torch.save(folds, out_dir / f'{name}_{sweep_identifier}.pt')


def _compute_estimators(
    split_lists: SplitLists,
    activations: dict[str, torch.Tensor],
    supervised_pair_masks_dict: dict[str, PairMasks],
    fold_ids_dict: dict[str, torch.Tensor],
    layer: int,
    pooling_type: PoolingTypeEnum,
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> Estimators:
    sweep_identifier = get_sweep_identifier(layer, pooling_type)
    difference_vectors = {}
    fold_mean_vectors = {}

    for pair in split_lists.supervised_pairs:
        aae_back_activation_matrix = None
        if layer == experiment_config.primary_layer:
            aae_back_split_name = f'{pair.corpus_name}{SplitSuffixes.AAE_BACK}{SplitSuffixes.SUPERVISED}'
            aae_back_activation_matrix = _load_activation_matrix(aae_back_split_name, sweep_identifier, run_dirs.activations)
        difference_matrices = _compute_difference_matrices(
            pair, activations, supervised_pair_masks_dict[pair.corpus_name], aae_back_activation_matrix,
        )
        for vector, difference_matrix in difference_matrices.items():
            difference_vectors[f'{DIFFERENCE_VECTOR_PREFIX}{pair.corpus_name}{vector}'] = math_utils.compute_mean_vector(difference_matrix)
            fold_mean_vectors[f'{Q1_FOLD_MEAN_VECTORS_PREFIX}{pair.corpus_name}{vector}'] = stats_utils.compute_fold_excluded_mean_vectors(
                difference_matrix,
                fold_ids_dict[pair.corpus_name],
            )

    estimators = Estimators(
        difference_vectors=difference_vectors,
        fold_mean_vectors=fold_mean_vectors,
    )
    return estimators


def compute_and_save_estimators(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    split_lists = _get_split_lists(corpus_config)
    max_rank = max(experiment_config.target_ranks)

    sweep_tuples = itertools.product(
        experiment_config.target_layers,
        PoolingTypeEnum,
    )
    supervised_pair_masks_dict = {
        pair.corpus_name: _load_supervised_pair_masks(pair, experiment_config, run_dirs.text_annotations, processed_data_dir)
        for pair in split_lists.supervised_pairs
    }
    fold_ids_dict = {
        corpus_name: stats_utils.make_fold_ids(len(pair_masks.aae_back_valid), experiment_config.num_cv_folds, experiment_config.random_seed)
        for corpus_name, pair_masks in supervised_pair_masks_dict.items()
    }
    for layer, pooling_type in tqdm(sweep_tuples, desc='Computing Estimators', leave=True):
        sweep_identifier = get_sweep_identifier(layer, pooling_type)
        split_names = (
            split_lists.supervised_split_names
            if pooling_type.is_paired_only
            else split_lists.all_eval_split_names
        )
        activations = {
            split_name: _load_activation_matrix(split_name, sweep_identifier, run_dirs.activations)
            for split_name in split_names
        }
        
        if layer == experiment_config.primary_layer and not pooling_type.is_paired_only:
            _compute_and_save_h1_tensors(activations, sweep_identifier, max_rank, experiment_config, run_dirs.h1_estimators)

        estimators = _compute_estimators(split_lists, activations, supervised_pair_masks_dict, fold_ids_dict, layer, pooling_type, experiment_config, run_dirs)
        _save_estimators(estimators.difference_vectors, layer, pooling_type, run_dirs.steering_estimators)
        _save_q1_folds(fold_ids_dict, estimators.fold_mean_vectors, sweep_identifier, run_dirs.q1_estimators)

        del activations, estimators


if __name__ == '__main__':
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    verify_extraction_log(EXPERIMENT_CONFIG, run_dirs)
    compute_and_save_estimators(CORPUS_CONFIG, EXPERIMENT_CONFIG, run_dirs)
    tqdm.write('04: Estimator Computation Complete\n')
