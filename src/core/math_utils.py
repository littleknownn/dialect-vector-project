from typing import NamedTuple

import torch
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BootstrapResults(NamedTuple):
    mean: float
    ci_lower: float
    ci_upper: float
    p_value: float


def compute_bootstrap(
    tensor: torch.Tensor,
    num_samples: int,
    bootstrap_alpha: float,
) -> BootstrapResults:
    tensor = tensor.view(-1)
    tensor_len = len(tensor)
    mean = tensor.mean().item()

    indices = torch.randint(tensor_len, size=(num_samples, tensor_len), device=tensor.device)
    sampled_tensor = tensor[indices]
    sampled_means = sampled_tensor.mean(dim=1)

    quantile = bootstrap_alpha / 2
    ci_lower = torch.quantile(sampled_means, quantile).item()
    ci_upper = torch.quantile(sampled_means, 1 - quantile).item()

    centered_sampled_means = sampled_means - mean
    significant_samples_mask = torch.abs(centered_sampled_means) >= abs(mean)
    p_value = significant_samples_mask.float().mean().item()

    results = BootstrapResults(
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
    )
    return results


def compute_chordal_mean_estimator(
        svd_estimators: list[torch.Tensor],
        rank: int,
) -> torch.Tensor:
    svd_estimators_range = torch.cat(svd_estimators, dim=1)
    U, _, _ = torch.linalg.svd(svd_estimators_range, full_matrices=False)
    return U[:, :rank].contiguous()


def compute_control_estimator(
    model_dim: int,
    rank: int,
    device: torch.device | str,
) -> torch.Tensor:
    gaussian_matrix = torch.randn(model_dim, rank, device=device)
    Q, _ = torch.linalg.qr(gaussian_matrix)
    return Q.contiguous()


def compute_leace_estimator(
        activation_matrices: list[torch.Tensor],
        rank: int,
) -> torch.Tensor:
    num_activations = sum(A.shape[0] for A in activation_matrices)
    sum_activation = sum(A.sum(dim=0) for A in activation_matrices)
    global_mean = sum_activation / num_activations

    centered_corpus_means = [A.mean(dim=0) - global_mean for A in activation_matrices]
    centered_corpus_means_matrix = torch.stack(centered_corpus_means, dim=0)

    actual_rank = min(rank, centered_corpus_means_matrix.shape[0])
    _, _, Vh = torch.linalg.svd(centered_corpus_means_matrix, full_matrices=False)
    V = Vh.T[:, :actual_rank]

    if actual_rank < rank:
        V = F.pad(V, pad=(0, rank - actual_rank))

    return V.contiguous()


def compute_supervised_estimator(
    aae_activation_matrix: torch.Tensor,
    sae_activation_matrix: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    difference_matrix = aae_activation_matrix - sae_activation_matrix
    _, _, V = torch.svd_lowrank(difference_matrix, q=rank)
    return V.contiguous()


def compute_vector_mean(
    activation_matrix: torch.Tensor,
) -> torch.Tensor:
    matrix_mean = activation_matrix.mean(dim=0)
    vector_mean = F.normalize(matrix_mean, dim=0).unsqueeze(1)
    return vector_mean


def compute_svd_estimator(
    activation_matrix: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    _, _, V = torch.pca_lowrank(activation_matrix, q=rank)
    return V.contiguous()
