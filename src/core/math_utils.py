import torch


def _get_valued_rows_mask(
    matrix: torch.Tensor,
) -> torch.Tensor:
    is_nan = matrix.isnan()
    valued_rows_mask = ~is_nan.all(dim=1)
    if (is_nan.any(dim=1) & valued_rows_mask).any():
        raise ValueError('Row should not be partially NaN')
    return valued_rows_mask


def compute_mean_vector(
    activation_matrix: torch.Tensor,
) -> torch.Tensor:
    valued_rows_mask = _get_valued_rows_mask(activation_matrix)
    mean_vector = activation_matrix[valued_rows_mask].mean(dim=0).unsqueeze(1)
    return mean_vector
