import torch

from src.core.utils import GeometricMetricsPair


def _get_valued_rows_mask(
    matrix: torch.Tensor,
) -> torch.Tensor:
    is_nan = matrix.isnan()
    valued_rows_mask = ~is_nan.all(dim=1)
    if (is_nan.any(dim=1) & valued_rows_mask).any():
        raise ValueError('Row should not be partially NaN')
    return valued_rows_mask


def compute_chordal_mean_estimator(
    svd_estimators: list[torch.Tensor],
    rank: int,
) -> torch.Tensor:
    svd_estimators_range = torch.cat(svd_estimators, dim=1)
    U, _, _ = torch.linalg.svd(svd_estimators_range, full_matrices=False)
    return U[:, :rank].contiguous()


def compute_matched_covariance_matrix(
    centered_matrix: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    matrix_len = centered_matrix.shape[0]
    random_matrix = torch.randn(matrix_len, rank, device=centered_matrix.device)

    directions_matrix = centered_matrix.T @ random_matrix / (matrix_len - 1) ** 0.5
    subspace_matrix, _ = torch.linalg.qr(directions_matrix)
    return subspace_matrix.contiguous()


def compute_vector_energy(
    estimators_tensor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    norm_vector = vector / torch.linalg.vector_norm(vector)
    subspace_projections = estimators_tensor.transpose(-1, -2) @ norm_vector
    vector_energies_tensor = (subspace_projections ** 2).sum(dim=(-1, -2))
    return vector_energies_tensor


def compute_mean_vector(
    activation_matrix: torch.Tensor,
) -> torch.Tensor:
    valued_rows_mask = _get_valued_rows_mask(activation_matrix)
    mean_vector = activation_matrix[valued_rows_mask].mean(dim=0).unsqueeze(1)
    return mean_vector


def compute_svd_estimator(
    activation_matrix: torch.Tensor,
    rank: int,
    oversample_rank: int = 10,
    n_iter: int = 4,
) -> torch.Tensor:
    _, _, V = torch.pca_lowrank(activation_matrix, q=oversample_rank, center=True, niter=n_iter)
    return V[:, :rank].contiguous()


def compute_geometric_metrics(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,    
    ) -> GeometricMetricsPair:
    product_tensor = tensor_a.transpose(-1, -2) @ tensor_b
    singular_values = torch.linalg.svdvals(product_tensor).clamp(max=1.0)
    
    principal_angles = torch.arccos(singular_values)
    chordal_distance = torch.sqrt(
        (tensor_a.shape[-1] - (singular_values ** 2).sum(dim=-1)).clamp(min=0.0)
    )

    metrics = GeometricMetricsPair(
        principal_angles=principal_angles,
        chordal_distance=chordal_distance,
    )
    return metrics
