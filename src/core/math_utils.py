import torch
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_svd_estimator(
        activation_matrix: torch.Tensor,
        rank: int,
) -> torch.Tensor:
    _, _, V = torch.pca_lowrank(activation_matrix, q=rank, center=True)
    return V.contiguous()

def compute_supervised_estimator(
    aae_activation_matrix: torch.Tensor,
    sae_activation_matrix: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    difference_matrix = aae_activation_matrix - sae_activation_matrix
    _, _, V = torch.svd_lowrank(difference_matrix, q=rank)
    return V.contiguous()

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
