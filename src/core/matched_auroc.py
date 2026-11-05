import math
from typing import NamedTuple

from scipy.stats import norm
import torch


NO_NEGATIVE_IDX: int = -1
NULL_SHUFFLE_CHUNK_SIZE: int = 500


class MatchedSets(NamedTuple):
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor


def _draw_negative_sets(
    negative_pool: torch.Tensor,
    num_positives: int,
    num_negatives: int,
    generator: torch.Generator,
    no_negative_idx: int = NO_NEGATIVE_IDX,
) -> torch.Tensor:
    negative_pool_len = len(negative_pool)
    negative_pool_shuffled = negative_pool[torch.randperm(negative_pool_len, generator=generator)]
    num_fresh_sets = min(num_positives, negative_pool_len // num_negatives)
    fresh_sets = negative_pool_shuffled[:num_fresh_sets * num_negatives].view(num_fresh_sets, num_negatives)

    num_reused_sets = num_positives - num_fresh_sets
    set_size = min(num_negatives, negative_pool_len)
    reused_set_positions = torch.rand(
        num_reused_sets, negative_pool_len, generator=generator,
    ).argsort(dim=1)[:, :set_size]
    reused_sets = torch.full((num_reused_sets, num_negatives), no_negative_idx)
    reused_sets[:, :set_size] = negative_pool[reused_set_positions]

    negative_sets = torch.cat([fresh_sets, reused_sets])
    return negative_sets


def draw_matched_sets(
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    cell_ids: torch.Tensor,
    num_negatives: int,
    seed: int,
) -> MatchedSets:
    generator = torch.Generator().manual_seed(seed)
    positive_indices_list, negative_indices_list = [], []
    for cell_id in cell_ids[positive_mask].unique().tolist():
        in_cell = cell_ids == cell_id
        positives = (positive_mask & in_cell).nonzero().squeeze(1)
        negatives = (negative_mask & in_cell).nonzero().squeeze(1)
        if len(negatives) == 0:
            continue
        positives_shuffled = positives[torch.randperm(len(positives), generator=generator)]
        positive_indices_list.append(positives_shuffled)
        negative_indices_list.append(_draw_negative_sets(
            negative_pool=negatives,
            num_positives=len(positives_shuffled),
            num_negatives=num_negatives,
            generator=generator,
        ))
    matched_sets = MatchedSets(
        positive_indices=torch.cat(positive_indices_list),
        negative_indices=torch.cat(negative_indices_list),
    )
    return matched_sets


def compute_per_positive_aurocs(
    values: torch.Tensor,
    matched_sets: MatchedSets,
    no_negative_idx: int = NO_NEGATIVE_IDX,
) -> torch.Tensor:
    is_valid = matched_sets.negative_indices != no_negative_idx
    positives = values[matched_sets.positive_indices].unsqueeze(1)
    negatives = values[matched_sets.negative_indices.clamp(min=0)]
    wins = (positives > negatives).double() + 0.5 * (positives == negatives).double()
    per_positive_aurocs = (wins * is_valid).sum(dim=1) / is_valid.sum(dim=1)
    return per_positive_aurocs


def compute_per_positive_differences(
    values: torch.Tensor,
    matched_sets: MatchedSets,
    no_negative_idx: int = NO_NEGATIVE_IDX,
) -> torch.Tensor:
    is_valid = matched_sets.negative_indices != no_negative_idx
    positives = values[matched_sets.positive_indices]
    negatives = values[matched_sets.negative_indices.clamp(min=0)]
    negative_means = (negatives * is_valid).sum(dim=1) / is_valid.sum(dim=1)
    per_positive_differences = positives - negative_means
    return per_positive_differences


def _compute_cell_null_auroc_sums(
    values: torch.Tensor,
    cell_members: torch.Tensor,
    num_positives: int,
    num_negatives: int,
    num_shuffles: int,
    generator: torch.Generator,
    null_shuffle_chunk_size: int = NULL_SHUFFLE_CHUNK_SIZE,
) -> torch.Tensor:
    num_needed = num_positives * (num_negatives + 1)
    if num_needed <= len(cell_members):
        auroc_sums = []
        for shuffle_chunk in torch.arange(num_shuffles).split(null_shuffle_chunk_size):
            permutations = torch.rand(
                len(shuffle_chunk), len(cell_members), generator=generator,
            ).argsort(dim=1)[:, :num_needed]
            shuffled_values = values[cell_members[permutations]]
            positive_values = shuffled_values[:, :num_positives].unsqueeze(2)
            negative_values = shuffled_values[:, num_positives:].view(len(shuffle_chunk), num_positives, num_negatives)
            wins = (positive_values > negative_values).double() + 0.5 * (positive_values == negative_values).double()
            auroc_sums.append(wins.mean(dim=2).sum(dim=1))
        return torch.cat(auroc_sums)

    auroc_sums = torch.empty(num_shuffles, dtype=torch.float64)
    for shuffle_idx in range(num_shuffles):
        shuffled_members = cell_members[torch.randperm(len(cell_members), generator=generator)]
        shuffled_sets = MatchedSets(
            positive_indices=shuffled_members[:num_positives],
            negative_indices=_draw_negative_sets(
                shuffled_members[num_positives:], num_positives, num_negatives, generator,
            ),
        )
        auroc_sums[shuffle_idx] = compute_per_positive_aurocs(values, shuffled_sets).sum()
    return auroc_sums


def compute_label_shuffle_null(
    values: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    cell_ids: torch.Tensor,
    num_negatives: int,
    num_shuffles: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    auroc_sums = torch.zeros(num_shuffles, dtype=torch.float64)
    num_positives_total = 0
    for cell_id in cell_ids[positive_mask].unique().tolist():
        in_cell = cell_ids == cell_id
        num_positives = int((positive_mask & in_cell).sum())
        if not(negative_mask & in_cell).any():
            continue
        cell_members = ((positive_mask | negative_mask) & in_cell).nonzero().squeeze(1)
        auroc_sums += _compute_cell_null_auroc_sums(
            values, cell_members, num_positives, num_negatives, num_shuffles, generator,
        )
        num_positives_total += num_positives
    null_aurocs = auroc_sums / num_positives_total
    return null_aurocs


def compute_minimum_detectable_auroc(
    num_positives: int,
    num_negatives: int,
    p_alpha: float,
    power: float,
    chance_auroc: float = 0.5,
) -> float:
    z_sum = norm.ppf(1 - p_alpha / 2) + norm.ppf(power)
    standard_error = math.sqrt((num_positives + num_negatives + 1) / (12 * num_positives * num_negatives)) #
    min_detectable_auroc = float(chance_auroc + z_sum * standard_error)
    return min_detectable_auroc
