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
    EXPERIMENT_CONFIG,
    ExperimentConfig,
)
from src.core.paths import (
    ACTIVATIONS_DIR,
    PROCESSED_DATA_DIR,
    get_sweep_identifier,
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
        tokens_batched_len = tokens['attention_mask'].sum(dim=1)
        token_indices_batched = torch.arange(len(batch), device=model.device)

        mean_pooling_mask = attention_mask.clone()
        if tokenizer.bos_token_id is not None:
            bos_mask = (tokens['input_ids'] == tokenizer.bos_token_id).unsqueeze(-1)
            mean_pooling_mask = mean_pooling_mask.masked_fill(bos_mask, 0)

        for layer in experiment_config.target_layers:
            layer_activations = outputs.hidden_states[layer + 1]
            last_activation = layer_activations[token_indices_batched, tokens_batched_len - 1, :]

            summed_activations = (layer_activations * mean_pooling_mask).sum(dim=1)
            mean_activation = summed_activations / mean_pooling_mask.sum(dim=1)

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


def extract_and_save_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    experiment_config: ExperimentConfig,
    in_dir: Path,
    out_dir: Path,
) -> None:
    model.eval()
    for corpus_path in in_dir.glob('*.jsonl'):
        split_name = corpus_path.stem

        activation_filepaths = (
            (out_dir / f'{split_name}_{get_sweep_identifier(layer, pooling_type)}.pt')
            for layer in experiment_config.target_layers
            for pooling_type in PoolingTypeEnum
        )
        if all(filepath.exists() for filepath in activation_filepaths):
            tqdm.write(f'Skipping {split_name} Preexisting Activations')
            continue

        tqdm.write(f'Loading {split_name} Corpus...')
        lines = load_jsonl(corpus_path)
        activations = _extract_activations(
            model,
            tokenizer,
            lines,
            experiment_config,
        )

        for layer, activation_matrices in tqdm(activations.items(), desc=f'Saving Activations'):
            last_identifier = get_sweep_identifier(layer, PoolingTypeEnum.LAST)
            mean_identifier = get_sweep_identifier(layer, PoolingTypeEnum.MEAN)

            last_filename = f'{split_name}_{last_identifier}.pt'
            mean_filename = f'{split_name}_{mean_identifier}.pt'

            last_path = out_dir / last_filename
            mean_path = out_dir / mean_filename

            torch.save(activation_matrices.last, last_path)
            torch.save(activation_matrices.mean, mean_path)

        del activations
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    model, tokenizer = load_base_model(EXPERIMENT_CONFIG)
    extract_and_save_activations(model, tokenizer, EXPERIMENT_CONFIG, PROCESSED_DATA_DIR, ACTIVATIONS_DIR)
    tqdm.write('02: Activation Extraction Complete\n')
