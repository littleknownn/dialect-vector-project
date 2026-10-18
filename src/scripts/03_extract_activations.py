from collections import defaultdict
import dataclasses
import gc
import hashlib
import itertools
from pathlib import Path
from typing import NamedTuple

import torch
from tqdm import tqdm
import transformers
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.base_model import load_base_model_and_tokenizer
from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
)
from src.core.config_utils import (
    DatasetConfig,
    ExperimentConfig,
    SupervisedDatasetConfig,
    get_dataset_config,
)
from src.core.paths import (
    EXTRACTION_LOG_FILENAME,
    PROCESSED_DATA_DIR,
    RunDirs,
    get_run_dirs,
    get_sweep_identifier,
    get_sweep_tensor_filepath,
)
from src.core.utils import (
    ExtractionLog,
    PoolingTypeEnum,
    SplitSuffixes,
    get_boundary_token_idx,
    load_boundary_char_indices_by_split,
    load_jsonl,
    save_json,
)


class ExtractedActivations(NamedTuple):
    activation_matrices_by_layer: dict[int, dict[PoolingTypeEnum, torch.Tensor]]
    token_states: list[torch.Tensor]


def _extract_activations(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        lines: list[str],
        layers: list[int],
        experiment_config: ExperimentConfig,
        boundary_char_indices: list[int | None] | None = None,
        save_token_states: bool = False,
    ) -> ExtractedActivations:
    token_states = []
    activations_by_layer = {
        layer: defaultdict(list) for layer in layers
    }
    line_boundary_char_indices = (
        boundary_char_indices
        if boundary_char_indices is not None
        else itertools.repeat(None)
    )
    batches = itertools.batched(zip(lines, line_boundary_char_indices), experiment_config.batch_size)
    for batch in tqdm(batches, desc='Extracting Activations'):
        batch_lines, batch_boundary_char_indices = zip(*batch)
        tokens = tokenizer(
            list(batch_lines),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=experiment_config.max_seq_len,
            return_offsets_mapping=True,
        )
        token_char_indices = tokens.pop('offset_mapping')
        tokens = tokens.to(model.device)

        with torch.inference_mode():
            outputs = model(**tokens, output_hidden_states=True)

        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        tokens_batched_len = tokens['attention_mask'].sum(dim=1)
        token_indices_batched = torch.arange(len(batch), device=model.device)

        mean_pooling_mask = attention_mask.clone()
        if tokenizer.bos_token_id is not None:
            bos_mask = (tokens['input_ids'] == tokenizer.bos_token_id).unsqueeze(-1)
            mean_pooling_mask = mean_pooling_mask.masked_fill(bos_mask, 0)

        if save_token_states:
            primary_layer_states = outputs.hidden_states[experiment_config.primary_layer + 1]
            token_states.extend(
                line_states[:num_tokens].cpu()
                for line_states, num_tokens in zip(primary_layer_states, tokens_batched_len)
            )

        if boundary_char_indices is not None:
            has_boundary = torch.tensor(
                [
                    boundary_char_idx is not None
                    for boundary_char_idx in batch_boundary_char_indices
                ],
                device=model.device,
            )
            boundary_token_indices = torch.tensor(
                [
                    get_boundary_token_idx(
                        line_token_char_indices,
                        boundary_char_idx,
                    )
                    if boundary_char_idx is not None else 0
                    for line_token_char_indices, boundary_char_idx
                    in zip(token_char_indices, batch_boundary_char_indices)
                ],
                device=model.device,
            )

        for layer in layers:
            layer_activations = outputs.hidden_states[layer + 1].float()
            last_activation = layer_activations[token_indices_batched, tokens_batched_len - 1, :]

            summed_activations = (layer_activations * mean_pooling_mask).sum(dim=1)
            mean_activation = summed_activations / mean_pooling_mask.sum(dim=1)

            activations_by_layer[layer][PoolingTypeEnum.LAST].append(last_activation.cpu())
            activations_by_layer[layer][PoolingTypeEnum.MEAN].append(mean_activation.cpu())

            if boundary_char_indices is not None:
                boundary_activation = layer_activations[token_indices_batched, boundary_token_indices, :]
                boundary_activation = torch.where(has_boundary.unsqueeze(-1), boundary_activation, torch.nan)
                activations_by_layer[layer][PoolingTypeEnum.BOUNDARY].append(boundary_activation.cpu())

    activation_matrices_by_layer = {
        layer: {
            pooling_type: torch.cat(batch_activations)
            for pooling_type, batch_activations in activations_by_pooling_type.items()
        }
        for layer, activations_by_pooling_type in activations_by_layer.items()
    }
    extracted_activations = ExtractedActivations(activation_matrices_by_layer, token_states)
    return extracted_activations


def _get_split_target_layers(
    split_name: str,
    experiment_config: ExperimentConfig,
) -> list[int]:
    primary_layer_analysis_suffixes = (
        SplitSuffixes.AAE_BACK,
        SplitSuffixes.RESERVE,
        SplitSuffixes.STRIPPED,
    )
    if any(suffix in split_name for suffix in primary_layer_analysis_suffixes):
        return [experiment_config.primary_layer]
    return experiment_config.target_layers


def _compute_num_truncated_lines(
    tokenizer: PreTrainedTokenizerBase,
    lines: list[str],
    max_seq_len: int,
) -> int:
    token_ids_by_line = tokenizer(lines)['input_ids']
    num_truncated_lines = sum(len(token_ids) > max_seq_len for token_ids in token_ids_by_line)
    return num_truncated_lines


def extract_and_save_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
    run_dirs: RunDirs,
    extraction_log_filename: str = EXTRACTION_LOG_FILENAME,
    in_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    model.eval()
    boundary_char_indices_by_split = load_boundary_char_indices_by_split(in_dir)
    num_lines_by_split = {}
    num_truncated_lines_by_split = {}
    for corpus_path in sorted(in_dir.glob('*.jsonl')):
        split_name = corpus_path.stem
        target_layers = _get_split_target_layers(split_name, experiment_config)
        tqdm.write(f'Loading {split_name} Corpus...')
        lines = load_jsonl(corpus_path)[:experiment_config.debug_max_split_lines]
        num_lines_by_split[split_name] = len(lines)
        num_truncated_lines_by_split[split_name] = _compute_num_truncated_lines(tokenizer, lines, experiment_config.max_seq_len)
        is_unpaired = not isinstance(get_dataset_config(split_name, corpus_config), SupervisedDatasetConfig)
        activations = _extract_activations(
            model,
            tokenizer,
            lines,
            target_layers,
            experiment_config,
            boundary_char_indices=boundary_char_indices_by_split.get(split_name),
            save_token_states=is_unpaired,
        )

        for layer, activation_matrices in tqdm(activations.activation_matrices_by_layer.items(), desc='Saving Activations'):
            for pooling_type, activation_matrix in activation_matrices.items():
                sweep_identifier = get_sweep_identifier(layer, pooling_type)
                filename = get_sweep_tensor_filepath(run_dirs.activations, sweep_identifier, split_name)
                torch.save(activation_matrix, filename)

        if is_unpaired:
            torch.save(
                {
                    'token_states': torch.cat(activations.token_states),
                    'num_tokens_by_line': torch.tensor([len(line_states) for line_states in activations.token_states]),
                },
                run_dirs.token_states / f'{split_name}.pt',
            )

        del activations
        gc.collect()
        torch.cuda.empty_cache()

    processed_data_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(in_dir.iterdir())
        if path.is_file()
    }

    extraction_log = ExtractionLog(
        model_hf_path=experiment_config.model_hf_path,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        experiment_config=dataclasses.asdict(experiment_config),
        processed_data_hashes=processed_data_hashes,
        num_lines_by_split=num_lines_by_split,
        num_truncated_lines_by_split=num_truncated_lines_by_split,
    )
    save_json(extraction_log, run_dirs.activations, extraction_log_filename)


if __name__ == '__main__':
    run_dirs = get_run_dirs(EXPERIMENT_CONFIG)
    model, tokenizer = load_base_model_and_tokenizer(EXPERIMENT_CONFIG)
    extract_and_save_activations(model, tokenizer, CORPUS_CONFIG, EXPERIMENT_CONFIG, run_dirs=run_dirs)
    tqdm.write('03: Activation Extraction Complete\n')
