import gc
import itertools
import json
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
    BaselineDatasetConfig,
    CORPUS_CONFIG,
    DatasetConfig,
    EXPERIMENT_CONFIG,
    ExperimentConfig,
    SupervisedDatasetConfig,
)
from src.core.paths import (
    ACTIVATIONS_DIR,
    get_sweep_suffix,
    PROCESSED_DATA_DIR,
)
from src.core.utils import (
    load_jsonl,
    PoolingTypeEnum,
)


class LayerActivationMatrices(NamedTuple):
    mean: torch.Tensor
    last: torch.Tensor


def _get_dataset_config(
        corpus_name: str,
        corpus_config: dict[str, DatasetConfig],
    ) -> DatasetConfig:
    if corpus_name in corpus_config:
        return corpus_config[corpus_name]
    for corpus_name_untagged, config in corpus_config.items():
        if corpus_name.startswith(corpus_name_untagged):
            return config


def _load_corpus_data(processed_data_dir: Path) -> dict[str, list[str]]:
    corpus_paths_iter = processed_data_dir.glob('*.jsonl')
    corpus_data = {}
    for corpus_path in corpus_paths_iter:
        split_name = corpus_path.stem
        corpus_lines = load_jsonl(corpus_path)
        corpus_data[split_name] = corpus_lines
    return corpus_data


def _compute_corpus_splits(
        corpus_data: dict[str, list[str]],
        corpus_config: dict[str, DatasetConfig],
        seed: int
    ) -> dict[str, list[str]]:
    min_corpus_len = min(
        len(lines)
        for split_name, lines in corpus_data.items()
        if not isinstance(_get_dataset_config(split_name, corpus_config), BaselineDatasetConfig)
    )
    permutation_generator = torch.Generator().manual_seed(seed)

    corpus_splits = {}
    permuted_indices_by_dataset = {}

    for split_name, lines in corpus_data.items():
        config = _get_dataset_config(split_name, corpus_config)
        config_key = id(config)
        if config_key not in permuted_indices_by_dataset:
            num_indices = len(lines)
            permuted_indices_by_dataset[config_key] = torch.randperm(num_indices, generator=permutation_generator).tolist()
        permuted_indices = permuted_indices_by_dataset[config_key]
        if isinstance(config, BaselineDatasetConfig):
            corpus_splits[split_name] = [lines[i] for i in permuted_indices]
        elif isinstance(config, SupervisedDatasetConfig):
            permuted_supervised_lines = [lines[i] for i in permuted_indices[:min_corpus_len]]
            permuted_testing_lines = [lines[i] for i in permuted_indices[min_corpus_len:min_corpus_len * 2]]

            corpus_splits[f'{split_name}{config.supervised_split_suffix}'] = permuted_supervised_lines
            corpus_splits[f'{split_name}{config.testing_split_suffix}'] = permuted_testing_lines
        else:
            corpus_splits[split_name] = [lines[i] for i in permuted_indices[:min_corpus_len]]
    return corpus_splits


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

    corpus_data = _load_corpus_data(PROCESSED_DATA_DIR)
    corpus_splits = _compute_corpus_splits(
        corpus_data,
        CORPUS_CONFIG,
        seed=EXPERIMENT_CONFIG.random_seed,
    )

    for split_name in corpus_splits:
        tqdm.write(f'Loading {split_name} Corpus...')
        lines = corpus_splits[split_name]
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
