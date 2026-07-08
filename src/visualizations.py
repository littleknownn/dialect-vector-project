import matplotlib.pyplot as plt

def training_loss_curve(
        adapter_name: str,
        training_steps: list[int],
        training_losses: list[int],
        figure_size: tuple[int, int] = (4, 4), 
    ) -> plt.Figure:
    fig,ax = plt.subplots(figsize=figure_size)
    ax.set_title(f'PEFT Loss on {adapter_name} Corpus')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.plot(training_steps, training_losses)
    return fig


def grassmann_distance_curve(
        tensor_type: str,
        distances_dict: dict[str, list[float]],
        baseline_distance: float | None,
        figure_size: tuple[int, int] = (4, 4),
        top_spacer_ratio: float = 1.2, 
    ) -> plt.Figure:
    num_layers = len(next(iter(distances_dict.values())))
    layer_indices = range(1, num_layers + 1)
    
    fig,ax = plt.subplots(figsize=figure_size)
    ax.set_title(f'Grassmann Distances between LoRA {tensor_type} Update Subspaces')
    ax.set_xlabel(r'Layer Index')
    ax.set_xlim(1, num_layers)
    ax.set_ylabel(r'Grassmann Distance')
    ax.set_ylim(0, baseline_distance * top_spacer_ratio)
    for adapter_names, distances in distances_dict.items():
        ax.plot(layer_indices, distances, label=adapter_names)
    
    if baseline_distance is not None:
        ax.axhline(y=baseline_distance, label='Random Orthogonal Baseline', linestyle='--')
    
    ax.legend()
    return fig


def singular_values_curve(
        layer_idx: int | str,
        tensor_type: str,
        singular_values_dict: dict[str, list[float]],
        figure_size: tuple[int, int] = (4, 4),
        alpha=0.4,
    ) -> plt.Figure:
    num_singular_values = len(next(iter(singular_values_dict.values())))
    singular_value_indices = range(1, num_singular_values + 1)

    fig,ax = plt.subplots(figsize=figure_size)
    ax.set_title(f'Singular Values of LoRA {tensor_type} Matrices\nLayer {layer_idx}')
    ax.set_xlabel(r'Singular Value Rank $i$')
    ax.set_ylabel(r'Singular Values $\sigma_i$')
    for adapter_name, singular_values in singular_values_dict.items():
        ax.plot(singular_value_indices, singular_values, label=adapter_name, alpha=alpha)
    ax.legend()
    return fig
