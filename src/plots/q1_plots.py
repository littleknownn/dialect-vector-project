import matplotlib.pyplot as plt
import torch

from src.core.utils import (
    PairEditData,
    Q1Results,
)
from src.plots.plot_utils import (
    DEFAULT_FIGSIZE,
    format_name,
)


def plot_projection_histogram(
    projections: torch.Tensor,
    q1_resutls: Q1Results,
    pair_edit_data: list[PairEditData] = False,
    ax: plt.Axes | None = None,
    num_bins: int = 60,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    group_masks = {'All Pairs': torch.ones_like(projections, dtype=torch.bool)}
    if pair_edit_data is not None:
        ending_changed_mask = torch.tensor([pair['ending_changed'] for pair in pair_edit_data])
        group_masks = {
            'Ending Changed': ending_changed_mask,
            'Ending Unchanged': ~ending_changed_mask,
        }
    bin_edges = torch.histogram(projections.float(), bins=num_bins).bin_edges.numpy()
    for label, mask in group_masks.items():
        ax.hist(
            projections[mask].numpy(),
            bins=bin_edges,
            label=f'{label} (n={mask.sum().item():,})', #
            alpha=0.5, #
        )    

    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5) #

    positive_ratio = q1_resutls['positive_projection_ratio']['estimate']
    standardized_mean = q1_resutls['standardized_projection_mean']
    summary = (
        f'Pos. Ratio: {positive_ratio:.2%}\n'
        f'Standardized Mean: {standardized_mean['estimate']:.2f}\n'
        f'CI: ({standardized_mean['ci_lower']:.2f}, {standardized_mean['ci_upper']:.2f})'
    )
    ax.text(0.02, 0.99, summary, transform=ax.transAxes, va='top', fontsize=8) #

    ax.set_title(f'{format_name(q1_resutls['pooling_type'])} Pooling')
    ax.set_xlabel('Projection onto held-out vector')
    ax.set_ylabel('Num. Pairs')
    ax.grid(axis='y', alpha=0.3)
    if pair_edit_data is not None:
        _, upper_ylim = ax.get_ylim()
        ax.set_ylim(top=upper_ylim * 1.2)
        ax.legend(loc='upper right')
    return ax