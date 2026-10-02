import gc
import itertools
from pathlib import Path

import torch
from tqdm import tqdm

from src.core.config import (
    CORPUS_CONFIG,
    DatasetConfig,
    ExperimentConfig,
    EXPERIMENT_CONFIG,
    get_estimator_data,
)
import src.core.math_utils as math_utils
from src.core.paths import (
    ACTIVATIONS_DIR,
    ESTIMATORS_DIR,
    get_sweep_identifier,
)
from src.core.utils import (
    AggregateEstimators,
    PoolingTypeEnum,
    SingleEstimatorPrefixes,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            out_filepath = out_dir / f'{estimator_name}_{sweep_identifier}.pt'
            torch.save(rank_estimator_matrix, out_filepath)
    act_add_sweep_identifier = get_sweep_identifier(layer, pooling_type, 1)
    for estimator_name, estimator_matrix in estimators.items():
        if estimator_name.startswith(SingleEstimatorPrefixes.ACT_ADD):
            out_filepath = out_dir / f'{estimator_name}_{act_add_sweep_identifier}.pt'
            torch.save(estimator_matrix, out_filepath)
        else:
            for vector_rank in experiment_config.target_act_add_ranks:
                if vector_rank > estimator_matrix.shape[1]:
                    continue
                vector = estimator_matrix[:, [vector_rank - 1]].contiguous()
                vector_name = f'{SingleEstimatorPrefixes.ACT_ADD}{vector_rank}_{estimator_name}'
                out_filepath = out_dir / f'{vector_name}_{act_add_sweep_identifier}.pt'
                torch.save(vector, out_filepath)


def compute_and_save_estimators(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    in_dir: Path,
    out_dir: Path,
) -> None:
    supervised_split_pairs, aae_corpora_splits = get_estimator_data(corpus_config)
    supervised_aae_splits = [pair.aae_split for pair in supervised_split_pairs]
    all_aae_splits = aae_corpora_splits + supervised_aae_splits
    max_rank = max(experiment_config.target_ranks)
    sweep_tuples = itertools.product(
        experiment_config.target_layers,
        PoolingTypeEnum,
    )
    for layer, pooling_type in tqdm(sweep_tuples, desc='Computing Estimators'):
        sweep_identifier = get_sweep_identifier(layer, pooling_type)
        model_dim = None
        estimators = {}
        for split_name in all_aae_splits:
            activation_matrix = torch.load(
                in_dir / f'{split_name}_{sweep_identifier}.pt',
                weights_only=True,
            ).to(device, dtype=torch.float32)

            estimators[f'{SingleEstimatorPrefixes.SVD}{split_name}'] = math_utils.compute_svd_estimator(
                activation_matrix,
                max_rank,
            )

            estimators[f'{SingleEstimatorPrefixes.ACT_ADD}{split_name}'] = math_utils.compute_vector_mean(
                activation_matrix,
            )
            
            if model_dim is None:
                model_dim = activation_matrix.shape[-1]
            del activation_matrix

        for split_pair in supervised_split_pairs:
            aae_activation_matrix = torch.load(
                in_dir / f'{split_pair.aae_split}_{sweep_identifier}.pt',
                weights_only=True,
            ).to(device, dtype=torch.float32)

            sae_activation_matrix = torch.load(
                in_dir / f'{split_pair.sae_split}_{sweep_identifier}.pt',
                weights_only=True,
            ).to(device, dtype=torch.float32)

            estimators[f'{SingleEstimatorPrefixes.SUPERVISED}{split_pair.corpus_name}'] = math_utils.compute_supervised_estimator(
                aae_activation_matrix,
                sae_activation_matrix,
                max_rank,
            )

            estimators[f'{SingleEstimatorPrefixes.SVD}{split_pair.sae_split}'] = math_utils.compute_svd_estimator(
                sae_activation_matrix,
                max_rank,
            )

            estimators[f'{SingleEstimatorPrefixes.ACT_ADD}{split_pair.corpus_name}'] = math_utils.compute_vector_mean(
                aae_activation_matrix - sae_activation_matrix,
            )

            estimators[f'{SingleEstimatorPrefixes.SVD}{split_pair.corpus_name}_half_1'] = math_utils.compute_svd_estimator(
                aae_activation_matrix[:len(aae_activation_matrix) // 2],
                max_rank,
            )

            estimators[f'{SingleEstimatorPrefixes.SVD}{split_pair.corpus_name}_half_2'] = math_utils.compute_svd_estimator(
                aae_activation_matrix[len(aae_activation_matrix) // 2:],
                max_rank,
            )

            del aae_activation_matrix, sae_activation_matrix

        aae_corpora_activations_list = [
            torch.load(
                in_dir / f'{split_name}_{sweep_identifier}.pt',
                weights_only=True,
            ).to(device, dtype=torch.float32)
            for split_name in aae_corpora_splits
        ]
        estimators[AggregateEstimators.LEACE_CORPORA] = math_utils.compute_leace_estimator(
            aae_corpora_activations_list,
            max_rank,
        )

        all_aae_splits_activations_list = [
            torch.load(
                in_dir / f'{split_name}_{sweep_identifier}.pt',
                weights_only=True,
            ).to(device, dtype=torch.float32)
            for split_name in all_aae_splits
        ]
        estimators[AggregateEstimators.LEACE_ALL] = math_utils.compute_leace_estimator(
            all_aae_splits_activations_list,
            max_rank,
        )

        aae_corpora_svd_estimator_list = [
            estimators[f'{SingleEstimatorPrefixes.SVD}{split_name}'] for split_name in aae_corpora_splits
        ]
        estimators[AggregateEstimators.CHORDAL_MEAN_CORPORA] = math_utils.compute_chordal_mean_estimator(
            aae_corpora_svd_estimator_list,
            max_rank,
        )

        all_aae_splits_svd_estimator_list = [
            estimators[f'{SingleEstimatorPrefixes.SVD}{split_name}'] for split_name in all_aae_splits
        ]
        estimators[AggregateEstimators.CHORDAL_MEAN_ALL] = math_utils.compute_chordal_mean_estimator(
            all_aae_splits_svd_estimator_list,
            max_rank,
        )

        estimators[AggregateEstimators.CONTROL] = math_utils.compute_control_estimator(
            model_dim,
            max_rank,
            device,
        )
        
        _save_estimators(
            estimators,
            layer,
            pooling_type,
            experiment_config,
            out_dir,
        )
        del estimators
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    compute_and_save_estimators(
        CORPUS_CONFIG,
        EXPERIMENT_CONFIG,
        in_dir=ACTIVATIONS_DIR,
        out_dir=ESTIMATORS_DIR,
    )
    tqdm.write('03: Estimator Computation Complete\n')
