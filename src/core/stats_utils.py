from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.core.utils import CIStats
import src.core.math_utils as math_utils


type GroupStatistic = Callable[[torch.Tensor, torch.Tensor, list[list[int]]], torch.Tensor]


def compute_bootstrap_draws(
    values: torch.Tensor,
    statistic: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    seed: int,
    batch_size: int = 500,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    num_values = len(values)
    draws = []
    for batch_start_idx in range(0, num_samples, batch_size):
        batch_len = min(batch_size, num_samples - batch_start_idx)
        indices = torch.randint(num_values, (batch_len, num_values), generator=generator)
        draws.append(statistic(values[indices]))
    bootstrap_draws = torch.cat(draws)
    return bootstrap_draws


def compute_positive_projection_ratio(
    projections: torch.Tensor,
) -> torch.Tensor:
    return (projections > 0).double().mean(dim=-1)


def compute_standardized_projection_mean(
    projections: torch.Tensor,
) -> torch.Tensor:
    return projections.mean(dim=-1) / projections.std(dim=-1)


def compute_percentile_ci(
    draws: torch.Tensor,
    p_alpha: float,
    estimate: float | None = None,
) -> CIStats:
    quantiles = torch.tensor([p_alpha / 2, 1 - p_alpha / 2], dtype=torch.float64, device=draws.device)
    ci_lower, ci_upper = torch.quantile(draws.double(), quantiles).tolist()
    if estimate is None:
        estimate = float(draws.double().mean())
    ci_stats = CIStats(estimate=estimate, ci_lower=ci_lower, ci_upper=ci_upper)
    return ci_stats


def compute_bootstrap_ci(
    values: torch.Tensor,
    statistic: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    num_draws: int,
    p_alpha: float,
    seed: int,
) -> CIStats:
    if statistic is None:
        statistic = lambda values: values.mean(dim=-1)
    draws = compute_bootstrap_draws(values, statistic, num_draws, seed)
    ci_stats = compute_percentile_ci(draws, p_alpha, estimate=float(statistic(values)))
    return ci_stats


def compute_regression_coefficients(
    feature_matrix: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    lstsq = torch.linalg.lstsq(feature_matrix, targets.unsqueeze(-1))
    coefficients = lstsq.solution.squeeze(-1)
    return coefficients


def compute_group_shares(
    feature_matrix: torch.Tensor,
    targets: torch.Tensor,
    group_feature_indices: list[list[int]],
) -> torch.Tensor:
    coefficients = compute_regression_coefficients(feature_matrix, targets)
    contributions = coefficients * feature_matrix.mean(dim=-2)
    group_contributions = torch.stack(
        [contributions[..., feature_indices].sum(dim=-1) for feature_indices in group_feature_indices],
        dim=-1,
    )
    group_shares = group_contributions / targets.mean(dim=-1, keepdim=True)
    return group_shares


def compute_r_squared(
    feature_matrix: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    coefficients = compute_regression_coefficients(feature_matrix, targets)
    predictions = (feature_matrix @ coefficients.unsqueeze(-1)).squeeze(-1)
    residual_sum_of_squares = ((targets - predictions) ** 2).sum(dim=-1)
    total_sum_of_squares = ((targets - targets.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    return 1 - residual_sum_of_squares / total_sum_of_squares


def compute_unique_r_squared(
    feature_matrix: torch.Tensor,
    targets: torch.Tensor,
    group_feature_indices: list[list[int]],
) -> torch.Tensor:
    full_r_squared = compute_r_squared(feature_matrix, targets)
    num_features = feature_matrix.shape[-1]
    unique_r_squared_list = []
    for feature_indices in group_feature_indices:
        held_in_features = [feature for feature in range(num_features) if feature not in feature_indices]
        reduced_r_squared = compute_r_squared(feature_matrix[..., held_in_features], targets)
        unique_r_squared_list.append(full_r_squared - reduced_r_squared)
    unique_r_squared = torch.stack(unique_r_squared_list, dim=-1)
    return unique_r_squared


def make_fold_ids(
    num_pairs: int,
    num_folds: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_pairs, generator=generator)

    fold_ids = torch.empty(num_pairs, dtype=torch.long)
    for fold, fold_indices in enumerate(torch.tensor_split(permutation, num_folds)):
        fold_ids[fold_indices] = fold
    return fold_ids


def compute_fold_excluded_mean_vectors(
    matrix: torch.Tensor,
    fold_ids: torch.Tensor,
) -> torch.Tensor:
    fold_ids = fold_ids.to(matrix.device)
    num_folds = int(fold_ids.max()) + 1
    vector_mean_folds_list = [
        math_utils.compute_mean_vector(matrix[fold_ids != fold_id])
        for fold_id in range(num_folds)
    ]
    vector_mean_folds = torch.cat(vector_mean_folds_list, dim=1)
    return vector_mean_folds


def compute_sign_flip_p_value(
    projection: torch.Tensor,
    num_flips: int,
    seed: int,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    sign_vectors = -1 + 2 * torch.randint(0, 2, (num_flips, len(projection)), generator=generator).double()
    projection_mean = projection.double().mean()
    sign_flipped_means = (sign_vectors * projection.double()).mean(dim=1)
    num_ge = int((sign_flipped_means >= projection_mean).sum())
    p_value = (1 + num_ge) / (1 + num_flips)
    return p_value


def compute_fold_projections(
    difference_matrix: torch.Tensor,
    fold_ids: torch.Tensor,
    mean_vector_folds: torch.Tensor,
) -> torch.Tensor:
    fold_ids = fold_ids.to(difference_matrix.device)
    unit_differences = F.normalize(mean_vector_folds, dim=0)
    projections = torch.empty(len(difference_matrix), device=difference_matrix.device)
    for fold_id in range(unit_differences.shape[1]):
        fold_mask = (fold_ids == fold_id)
        projections[fold_mask] = difference_matrix[fold_mask] @ unit_differences[:, fold_id]
    return projections
