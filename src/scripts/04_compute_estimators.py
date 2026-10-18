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
    Q1_FOLD_IDS_PREFIX,
    Q1_FOLD_MEAN_VECTORS_PREFIX,
    H1_FLOORS_PREFIX,
    H1_NULL_PREFIX,
    H1_SVD_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
import src.core.stats_utils as stats_utils
from src.core.utils import (
    PoolingTypeEnum,
    SingleEstimatorPrefixes,
    save_json,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SplitLists(NamedTuple):
    supervised_pairs: list[SupervisedPair]
    supervised_split_names: list[str]
    aae_corpora_split_names: list[str]
    all_eval_split_names: list[str]


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
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    for rank in experiment_config.target_ranks:
        for estimator_name, estimator_matrix in estimators.items():
            if estimator_name.startswith(SingleEstimatorPrefixes.ACT_ADD):
                continue
            if rank > estimator_matrix.shape[1]:
                continue
            rank_estimator_matrix = estimator_matrix[:, :rank].contiguous()
            sweep_identifier = get_sweep_identifier(layer, pooling_type, rank)
            torch.save(
                rank_estimator_matrix,
                get_sweep_tensor_filepath(out_dir, sweep_identifier, estimator_name),
            )
    act_add_sweep_identifier = get_sweep_identifier(layer, pooling_type, 1)
    for estimator_name, estimator_matrix in estimators.items():
        if estimator_name.startswith(SingleEstimatorPrefixes.ACT_ADD):
            torch.save(
                estimator_matrix,
                get_sweep_tensor_filepath(out_dir, act_add_sweep_identifier, estimator_name),
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


def _compute_pair_mean_difference_vector(
    pairs: list[SupervisedPair],
    activations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    estimators = {}
    for pair in pairs:
        aae_activation_matrix = activations[pair.aae_split]
        sae_activation_matrix = activations[pair.sae_split]
        estimators[f'{SingleEstimatorPrefixes.ACT_ADD}{pair.corpus_name}'] = math_utils.compute_mean_vector(
            aae_activation_matrix - sae_activation_matrix,
        )
    return estimators


def _compute_and_save_q1_folds(
    supervised_pairs: list[SupervisedPair],
    activations: dict[str, torch.Tensor],
    sweep_identifier: str,
    experiment_config: ExperimentConfig,
    out_dir: Path,
) -> None:
    for pair in supervised_pairs:
        difference_matrix = activations[pair.aae_split] - activations[pair.sae_split]
        fold_ids = stats_utils.make_fold_ids(len(difference_matrix), experiment_config.num_cv_folds, experiment_config.random_seed)
        fold_mean_vectors = stats_utils.compute_fold_excluded_mean_vectors(difference_matrix, fold_ids)

        torch.save(fold_ids, out_dir / f'{Q1_FOLD_IDS_PREFIX}{pair.corpus_name}.pt')
        torch.save(fold_mean_vectors, out_dir / f'{Q1_FOLD_MEAN_VECTORS_PREFIX}{pair.corpus_name}_{sweep_identifier}.pt')


def _compute_h3_estimators(
    aae_corpora_split_names: list[str],
    activations: dict[str, torch.Tensor],
    max_rank: int,
    random_seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(random_seed)
    split_svds = {
        split_name: math_utils.compute_svd_estimator(activations[split_name], max_rank)
        for split_name in aae_corpora_split_names
    }
    estimators = {}
    for held_out_split in aae_corpora_split_names:
        used_splits = [split_name for split_name in aae_corpora_split_names if split_name != held_out_split]
        estimators[f'{SingleEstimatorPrefixes.H3_CONSENSUS}{held_out_split}'] = math_utils.compute_chordal_mean_estimator(
            svd_estimators=[split_svds[split_name] for split_name in used_splits],
            rank=max_rank,
        )
        used_activations_centered = torch.cat([
            activations[split_name] - activations[split_name].mean(dim=0)
            for split_name in used_splits
        ])
        estimators[f'{SingleEstimatorPrefixes.H3_POOLED}{held_out_split}'] = math_utils.compute_svd_estimator(
            activation_matrix=used_activations_centered,
            rank=max_rank,
        )

    return estimators


def compute_and_save_estimators(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
) -> None:
    split_lists = _get_split_lists(corpus_config)
    max_rank = max(experiment_config.target_ranks)

    sweep_tuples = itertools.product(
        experiment_config.target_layers,
        PoolingTypeEnum,
    )
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
        _compute_and_save_q1_folds(split_lists.supervised_pairs, activations, sweep_identifier, experiment_config, run_dirs.q1_estimators)

        estimators = _compute_pair_mean_difference_vector(split_lists.supervised_pairs, activations)
        if not pooling_type.is_paired_only:
            estimators |= _compute_h3_estimators(split_lists.aae_corpora_split_names, activations, max_rank, random_seed=experiment_config.random_seed)
        _save_estimators(estimators, layer, pooling_type, experiment_config, run_dirs.steering_estimators)

        del activations, estimators


if __name__ == '__main__':
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    compute_and_save_estimators(CORPUS_CONFIG, EXPERIMENT_CONFIG, run_dirs)
    tqdm.write('04: Estimator Computation Complete\n')
