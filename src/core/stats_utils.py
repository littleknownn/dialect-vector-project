from collections.abc import Callable

from scipy.stats import rankdata
import torch
import torch.nn.functional as F

from src.core.utils import CIStats
import src.core.math_utils as math_utils


def _compute_ranks(
    values: torch.Tensor,
) -> torch.Tensor:
    ranks = torch.from_numpy(rankdata(values.numpy(), axis=-2))
    return ranks


def _compute_correlation(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> torch.Tensor:
    centered_a = tensor_a - tensor_a.mean(dim=-1, keepdim=True)
    centered_b = tensor_b - tensor_b.mean(dim=-1, keepdim=True)
    correlation = (centered_a * centered_b).sum(dim=-1) / (
        torch.linalg.vector_norm(centered_a, dim=-1)
        * torch.linalg.vector_norm(centered_b, dim=-1)
    )
    return correlation


def compute_spearman(
    values: torch.Tensor,
) -> torch.Tensor:
    ranks = _compute_ranks(values)
    spearman = _compute_correlation(ranks[..., 0], ranks[..., 1])
    return spearman


def compute_partial_spearman(
    values: torch.Tensor,
) -> torch.Tensor:
    ranks = _compute_ranks(values)
    design_matrix = torch.stack([torch.ones_like(ranks[..., 2]), ranks[..., 2]], dim=-1)
    residualize_fn = lambda column : column - (
        design_matrix @ compute_regression_coefficients(design_matrix, column).unsqueeze(-1)
    ).squeeze(-1)
    partial_spearman = _compute_correlation(
        residualize_fn(ranks[..., 0]),
        residualize_fn(ranks[..., 1]),
    )
    return partial_spearman


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


def compute_r_squared(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    residual_sum_of_squares = (targets - predictions).square().sum(dim=-1)
    total_sum_of_squares = (targets - targets.mean(dim=-1, keepdim=True)).square().sum(dim=-1)
    r_squared = 1 - residual_sum_of_squares / total_sum_of_squares
    return r_squared


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


def compute_calibration(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int,
) -> list[list[float]]:
    bin_edges = torch.quantile(
        predictions,
        torch.linspace(0, 1, num_bins + 1, dtype=predictions.dtype)[1:-1],
    )
    bin_ids = torch.searchsorted(bin_edges, predictions)
    calibration = [
        [
            float(predictions[bin_ids == bin_id].mean()),
            float(targets[bin_ids == bin_id].mean()),
        ]
        for bin_id in range(num_bins)
    ]
    return calibration
