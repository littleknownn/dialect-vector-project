import itertools
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
    SupervisedPair,
    get_estimator_data,
)
import src.core.math_utils as math_utils
from src.core.paths import (
    H1_FLOORS_PREFIX,
    H1_NULL_PREFIX,
    H1_RESULTS_FILENAME,
    H1_SVD_PREFIX,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
from src.core.utils import (
    H1Results,
    PoolingTypeEnum,
    SingleEstimatorPrefixes,
    load_json,
    save_json,
)


class H1Data(NamedTuple):
    svd_tensors: dict[str, torch.Tensor]
    nulls: torch.Tensor
    floors: dict[str, dict[str, list[float]]]


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
    steering_estimators_dir: Path,
    supervised_corpus_name: str,
    estimator_rank: int,
) -> H1Results:
    act_add_sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type, 1)
    supervised_act_add_vector = torch.load(
        get_sweep_tensor_filepath(
            steering_estimators_dir,
            act_add_sweep_identifier,
            f'{SingleEstimatorPrefixes.ACT_ADD}{supervised_corpus_name}',
        ),
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
    run_dirs: RunDirs,
    h1_results_filename: str,
) -> None:
    max_rank = max(experiment_config.target_ranks)
    data = _load_h1_data(pooling_type, experiment_config, run_dirs.h1_estimators)
    results = _compute_h1_results(data, pooling_type, experiment_config, run_dirs.steering_estimators, pair.corpus_name, max_rank)
    sweep_identifier = get_sweep_identifier(experiment_config.primary_layer, pooling_type, max_rank)
    filename = f'{h1_results_filename}_{pair.corpus_name}_{sweep_identifier}'
    save_json(results, run_dirs.transfer_results, filename)


### Main ###
def compute_and_save_metrics(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    h1_results_filename: str = H1_RESULTS_FILENAME,
) -> None:
    supervised_pairs = get_estimator_data(corpus_config).supervised_split_pairs
    for pair, pooling_type in itertools.product(supervised_pairs, PoolingTypeEnum):
        if not pooling_type.is_paired_only:
            _compute_and_save_h1_results(pair, pooling_type, experiment_config, run_dirs, h1_results_filename)


if __name__ == '__main__':
    tqdm.write('Loading Data...')
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    compute_and_save_metrics(CORPUS_CONFIG, EXPERIMENT_CONFIG, run_dirs)
    tqdm.write('07: Transfer Metrics Computation Complete\n')
