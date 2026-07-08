from src.config import (
    ExperimentConfig,
    DatasetConfig,
    experiment_config,
    corpus_config,
)
from src.utils import (
    MetricsJson,
    save_json,
    models_dir,
    weights_filename,
    steering_vectors_dir,
    steering_vectors_filename,
    metrics_dir,
    phase_1_metrics_filename,
)
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoConfig
import itertools


type TensorsDict = dict[
    str, dict[
        int, dict[
            str, dict[
                str, torch.Tensor
            ]
        ]
    ]
]


def get_tensors_dict(
        corpus_config: dict[str, DatasetConfig],
        out_dir: Path = models_dir,
        out_filename: str = weights_filename,
    ) -> TensorsDict:
    tensors_dict = {}
    for corpus_name in corpus_config:
        tensors_path = out_dir / corpus_name / f'{out_filename}.pt'
        tensors_dict[corpus_name] = torch.load(tensors_path, weights_only=True)
    return tensors_dict


def compute_svd_results(
        experiment_config: ExperimentConfig,
        tensors_dict: TensorsDict,
        corpus_name: str,
        layer_idx: int,
        tensor_type: str,
    ) -> tuple[list[float], torch.Tensor]:
    lora_update_matrices = tensors_dict[corpus_name][layer_idx][tensor_type]
    A, B = lora_update_matrices['A'].float(), lora_update_matrices['B'].float()
    delta_W = B @ A
        
    U, S, _ = torch.linalg.svd(delta_W)
    singular_values = S[:experiment_config.lora_rank].tolist()
    left_singular_vectors = U[:, :experiment_config.lora_rank]
    return singular_values, left_singular_vectors


def compute_grassmann_distance(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    M = U_a.T @ U_b
    _, S, _ = torch.linalg.svd(M)
    principal_angles = torch.arccos(torch.clamp(S, min=-1.0, max=1.0))
    grassmann_distance = torch.norm(principal_angles).item()
    return grassmann_distance


def compute_chordal_mean(U_matrices:  list[torch.Tensor], rank: int) -> torch.Tensor:
    projection_matrices = [U @ U.T for U in U_matrices]
    mean_projection = torch.stack(projection_matrices).mean(dim=0)
    U, _, _ = torch.linalg.svd(mean_projection)
    chordal_mean = U[:, :rank]
    return chordal_mean


def compute_baseline_distance(
        experiment_config: ExperimentConfig,
        hidden_dim: int,
        num_trials: int=100
    ) -> float:
    distance = 0.0
    for _ in range(num_trials):
        A = nn.init.orthogonal_(torch.empty(hidden_dim, experiment_config.lora_rank))
        B = nn.init.orthogonal_(torch.empty(hidden_dim, experiment_config.lora_rank))
        distance += compute_grassmann_distance(A, B)
    avg_distance = distance / num_trials
    return avg_distance


def get_phase_1_metrics(
        experiment_config: ExperimentConfig,
        tensors_dict: TensorsDict,
        corpus_config: dict[str, DatasetConfig] = corpus_config,
    ) -> MetricsJson:
    metrics = {}
    layer_indices = sorted(next(iter(tensors_dict.values())).keys())

    for layer_idx in layer_indices:
        layer_metrics = {}
        metrics[str(layer_idx)] = layer_metrics

        for tensor_type in experiment_config.target_weight_matrices:
            corpus_names = corpus_config.keys()
            left_singular_vectors_lookup = {}
            layer_metrics[tensor_type] = {
                  'singular_values': {},
                  'grassmann_distances': {},
                }

            for corpus_name in corpus_names:
                singular_values, left_singular_vectors = compute_svd_results(
                    experiment_config,
                    tensors_dict,
                    corpus_name,
                    layer_idx,
                    tensor_type,
                )
                layer_metrics[tensor_type]['singular_values'][corpus_name] = singular_values
                left_singular_vectors_lookup[corpus_name] = left_singular_vectors
            
            for corpus_a, corpus_b in itertools.combinations(corpus_names, 2):
                U_a = left_singular_vectors_lookup[corpus_a]
                U_b = left_singular_vectors_lookup[corpus_b]
                grassmann_distance = compute_grassmann_distance(U_a, U_b)

                pair_name = f'{corpus_a}_vs_{corpus_b}'
                layer_metrics[tensor_type]['grassmann_distances'][pair_name] = grassmann_distance
    
    metrics['metadata'] = {'baseline': {}}
    for tensor_type in experiment_config.target_weight_matrices:
        sample_corpus = next(iter(corpus_config.keys()))
        sample_layer = next(iter(tensors_dict[sample_corpus].keys()))
        sample_tensor = next(iter(tensors_dict[sample_corpus][sample_layer][tensor_type].values()))
        hidden_dim = (
            sample_tensor.shape[0]
            if sample_tensor.shape[0] != experiment_config.lora_rank
            else sample_tensor.shape[1]
        )

        metrics['metadata']['baseline'][tensor_type] = compute_baseline_distance(experiment_config, hidden_dim)

    return metrics


def get_steering_vector(
        experiment_config: ExperimentConfig,
        corpus_config: dict[str, DatasetConfig],
        tensors_dict: TensorsDict,
        layer_idx: int,
        tensor_type: str,
    ) -> torch.Tensor:
    U_matrices = []
    for corpus_name in corpus_config:
        _, left_singular_vectors = compute_svd_results(
            experiment_config,
            tensors_dict,
            corpus_name,
            layer_idx,
            tensor_type,
        )
        U_matrices.append(left_singular_vectors)
    steering_vector = compute_chordal_mean(U_matrices, rank=1).squeeze()
    return steering_vector


def get_steering_vectors_dict(
        experiment_config: ExperimentConfig,
        corpus_config: dict[str, DatasetConfig],
        tensors_dict: TensorsDict,
    ) -> dict[int, dict[str, torch.Tensor]]:
    layer_indices = sorted(next(iter(tensors_dict.values())).keys())
    steering_vectors = {}
    for layer_idx in layer_indices:
        steering_vectors[layer_idx] = {}
        for tensor_type in experiment_config.target_weight_matrices:
            steering_vector = get_steering_vector(
                experiment_config,
                corpus_config,
                tensors_dict,
                layer_idx,
                tensor_type,
            )
            steering_vectors[layer_idx][tensor_type] = steering_vector
    return steering_vectors


if __name__ == '__main__':
    tensors_dict = get_tensors_dict(corpus_config)

    steering_vectors = get_steering_vectors_dict(experiment_config, corpus_config, tensors_dict)
    steering_vectors_filepath = steering_vectors_dir / f'{steering_vectors_filename}.pt'
    torch.save(steering_vectors, steering_vectors_filepath)

    metrics = get_phase_1_metrics(experiment_config, tensors_dict)
    save_json(metrics, metrics_dir, phase_1_metrics_filename)
    print('Phase 1 Subspace Analysis Complete')
