import gc
import itertools
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.base_model import load_base_model
from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
    BaselineDatasetConfig,
    ExperimentConfig,
    get_dataset_config,
)
from src.core.paths import (
    ACTIVATIONS_DIR,
    PROCESSED_DATA_DIR,
    get_sweep_suffix,
)
from src.core.utils import (
    PoolingTypeEnum,
    load_jsonl,
)


class LayerActivationMatrices(NamedTuple):
    mean: torch.Tensor
    last: torch.Tensor


def _extract_activations(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        lines: list[str],
        experiment_config: ExperimentConfig,
    ) -> dict[int, LayerActivationMatrices]:

    activations_by_layer = {
        layer: {
            PoolingTypeEnum.LAST: [],
            PoolingTypeEnum.MEAN: []
        }
        for layer in experiment_config.target_layers
    }

    batches = itertools.batched(lines, experiment_config.batch_size)
    for batch in tqdm(batches, desc='Extracting Activations'):
        tokens = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=experiment_config.max_seq_len,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model(**tokens, output_hidden_states=True)

        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        num_tokens_batched = tokens['attention_mask'].sum(dim=1)
        token_indices_batched = torch.arange(len(batch), device=model.device)

        for layer in experiment_config.target_layers:
            layer_activations = outputs.hidden_states[layer]
            
            last_activation = layer_activations[token_indices_batched, num_tokens_batched - 1, :]
            summed_activations = (layer_activations * attention_mask).sum(dim=1)
            mean_activation = summed_activations / num_tokens_batched.unsqueeze(-1)

            activations_by_layer[layer][PoolingTypeEnum.LAST].append(last_activation.cpu())
            activations_by_layer[layer][PoolingTypeEnum.MEAN].append(mean_activation.cpu())

    activation_matrices_by_layer = {
        layer: LayerActivationMatrices(
            last = torch.cat(activations_dict[PoolingTypeEnum.LAST]),
            mean = torch.cat(activations_dict[PoolingTypeEnum.MEAN]),
        )
        for layer, activations_dict in activations_by_layer.items()
    }

    return activation_matrices_by_layer


if __name__ == '__main__':
    model, tokenizer = load_base_model(EXPERIMENT_CONFIG)
    model.eval()

    for corpus_path in PROCESSED_DATA_DIR.glob('*.jsonl'):
        split_name = corpus_path.stem

        config = get_dataset_config(split_name, CORPUS_CONFIG)
        if isinstance(config, BaselineDatasetConfig):
            tqdm.write(f'Skipping {split_name} Baseline Corpus')
            continue

        # # temp
        # checked_path = ACTIVATIONS_DIR / f'{split_name}_layer{EXPERIMENT_CONFIG.target_layers[0]:02d}_{PoolingTypeEnum.MEAN}.pt'
        # if checked_path.exists():
        #     continue

        tqdm.write(f'Loading {split_name} Corpus...')
        lines = load_jsonl(corpus_path)
        activations = _extract_activations(
            model,
            tokenizer,
            lines,
            EXPERIMENT_CONFIG,
        )

        for layer, activation_matrices in tqdm(activations.items(), desc=f'Saving Activations'):
            last_suffix = get_sweep_suffix(layer, PoolingTypeEnum.LAST)
            mean_suffix = get_sweep_suffix(layer, PoolingTypeEnum.MEAN)

            last_filename = f'{split_name}{last_suffix}.pt'
            mean_filename = f'{split_name}{mean_suffix}.pt'

            last_path = ACTIVATIONS_DIR / last_filename
            mean_path = ACTIVATIONS_DIR / mean_filename

            torch.save(activation_matrices.last, last_path)
            torch.save(activation_matrices.mean, mean_path)

        del activations
        gc.collect()
        torch.cuda.empty_cache()
