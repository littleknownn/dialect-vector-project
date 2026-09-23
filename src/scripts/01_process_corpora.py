import csv
import re
from typing import Optional

from datasets import load_dataset
import requests
from tqdm import tqdm
import torch

from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
    BaselineDatasetConfig,
    CoraalSourceConfig,
    DatasetConfig,
    GithubSourceConfig,
    HFSourceConfig,
    RealAaeSourceConfig,
    SupervisedDatasetConfig,
    get_dataset_config,
)
from src.core.paths import PROCESSED_DATA_DIR
from src.core.utils import (
    check_text_len_bounds,
    save_jsonl,
)


def _clean_text(text: str, config: DatasetConfig) -> Optional[str]:
    text_lowercase = text.lower()
    for pattern in config.skipped_data_patterns:
        if re.search(pattern, text_lowercase):
            return None    
    
    for pattern, replacement in config.replacement_patterns:
        text = re.sub(pattern, replacement, text)

    if not (cleaned_text := text.strip()):
        return None
    
    for pattern in config.skipped_case_patterns:
        if re.search(pattern, cleaned_text):
            return None
    
    return cleaned_text


def _process_coraal_lines(lines: list[str], config: DatasetConfig) -> list[str]:
    utterances = []
    current_utterance = []
    current_speaker = None

    def flush_utterance():
        if current_utterance:
            combined_utterance = ' '.join(current_utterance)
            if (clean_utterance := _clean_text(combined_utterance, config)):
                utterances.append(clean_utterance)
            current_utterance.clear()

    for line in lines:
        cols = line.split(config.source_config.delimiter)
        if len(cols) <= config.source_config.text_col_idx:
            continue
        speaker = cols[config.source_config.speaker_col_idx]
        text = cols[config.source_config.text_col_idx]

        if config.source_config.interviewer_marker in speaker:
            flush_utterance()
            current_speaker = None
        
        elif current_speaker is None or current_speaker == speaker:
            current_speaker = speaker
            current_utterance.append(text)
        
        elif current_utterance:
            flush_utterance()
            current_speaker = speaker
            current_utterance.append(text)
    
    flush_utterance()
    return utterances


def process_coraal_dataset(config: DatasetConfig) -> list[str]:
    cleaned_lines = []
    for city_dir in tqdm(config.source_config.in_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for text_path in list(city_dir.glob('*.txt')):
            if text_path.name.startswith('._'):
                continue
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                lines = text_file.readlines()
                lines = lines[1:] if config.source_config.header else lines
                cleaned_lines.extend(_process_coraal_lines(lines, config))
    return cleaned_lines


def process_realaae_dataset(config: DatasetConfig) -> tuple[list[str], list[str]]:
    cleaned_lines_aae, cleaned_lines_sae = [], []
    dataset_path = next(config.source_config.in_dir.glob('*.csv'))
    with open(dataset_path, mode='r', encoding='utf-8', errors='ignore') as dataset_file:
        csv_reader = csv.reader(dataset_file, delimiter=config.source_config.delimiter)
        if config.source_config.header:
            next(csv_reader)
        for row in csv_reader:
            clean_row_aae = _clean_text(row[config.source_config.aae_col_idx], config)
            clean_row_sae = _clean_text(row[config.source_config.sae_col_idx], config)
            if (
                clean_row_aae and clean_row_sae
                and check_text_len_bounds(clean_row_aae, config.min_len_chars, config.min_len_words, config.max_len_words)
                and check_text_len_bounds(clean_row_sae, config.min_len_chars, config.min_len_words, config.max_len_words)
            ):
                cleaned_lines_aae.append(clean_row_aae)
                cleaned_lines_sae.append(clean_row_sae)
    return cleaned_lines_aae, cleaned_lines_sae


def process_hf_dataset(
        config: DatasetConfig,
        excluded_lines: set[str] | None = None,
) -> list[str]:
    dataset = load_dataset(config.source_config.hf_path, name=config.source_config.hf_name, split=config.source_config.split)
    excluded_lines = excluded_lines if excluded_lines else set()
    cleaned_lines = [
        clean_row
        for row in tqdm(dataset)
        if (clean_row := _clean_text(row[config.source_config.text_col], config))
        and clean_row.lower() not in excluded_lines
    ]
    return cleaned_lines


def process_gh_dataset(config: DatasetConfig) -> list[str]:
    lyrics = requests.get(config.source_config.git_url)
    lyrics.raise_for_status()
    lines = lyrics.text.split(config.source_config.delimiter)
    cleaned_lines = [
        cleaned_line
        for line in tqdm(lines)
        if (cleaned_line := _clean_text(line, config))
    ]
    return cleaned_lines


def _compute_corpus_splits(
        corpus_data: dict[str, list[str]],
        corpus_config: dict[str, DatasetConfig],
        seed: int
    ) -> dict[str, list[str]]:
    min_corpus_len = min(
        len(lines)
        for split_name, lines in corpus_data.items()
        if not isinstance(get_dataset_config(split_name, corpus_config), BaselineDatasetConfig)
    )
    permutation_generator = torch.Generator().manual_seed(seed)

    corpus_splits = {}
    permuted_indices_by_dataset = {}

    for split_name, lines in corpus_data.items():
        config = get_dataset_config(split_name, corpus_config)
        config_key = id(config)
        if config_key not in permuted_indices_by_dataset:
            num_indices = len(lines)
            permuted_indices_by_dataset[config_key] = torch.randperm(num_indices, generator=permutation_generator).tolist()
        permuted_indices = permuted_indices_by_dataset[config_key]

        if isinstance(config, BaselineDatasetConfig):
            corpus_splits[split_name] = [lines[i] for i in permuted_indices[:min_corpus_len]]
        elif isinstance(config, SupervisedDatasetConfig):
            permuted_supervised_lines = [lines[i] for i in permuted_indices[:min_corpus_len]]
            permuted_testing_lines = [lines[i] for i in permuted_indices[min_corpus_len:min_corpus_len * 2]]

            corpus_splits[f'{split_name}{config.supervised_split_suffix}'] = permuted_supervised_lines
            corpus_splits[f'{split_name}{config.testing_split_suffix}'] = permuted_testing_lines
        else:
            corpus_splits[split_name] = [lines[i] for i in permuted_indices[:min_corpus_len]]
    return corpus_splits



if __name__ == '__main__':
    real_aae_name, real_aae_config = next(
        (
            (corpus_name, config)
            for corpus_name, config in CORPUS_CONFIG.items()
            if isinstance(config.source_config, RealAaeSourceConfig)
        ),
        (None, None),
    )

    cleaned_corpus_data = {}
    twitter_aae_excluded_lines = set()
    
    if real_aae_name:
        cleaned_aae, cleaned_sae = process_realaae_dataset(real_aae_config)
        twitter_aae_excluded_lines = {line.lower() for line in cleaned_aae}

        cleaned_corpus_data[f'{real_aae_name}{real_aae_config.aae_split_suffix}'] = cleaned_aae
        cleaned_corpus_data[f'{real_aae_name}{real_aae_config.sae_split_suffix}'] = cleaned_sae

    for corpus_name, config in CORPUS_CONFIG.items():
        tqdm.write(f'Processing {corpus_name} Data...')

        if isinstance(config.source_config, RealAaeSourceConfig):
            continue
        elif isinstance(config.source_config, CoraalSourceConfig):
            cleaned_lines = process_coraal_dataset(config)
        elif isinstance(config.source_config, HFSourceConfig):
            cleaned_lines = process_hf_dataset(config, excluded_lines=twitter_aae_excluded_lines)
        elif isinstance(config.source_config, GithubSourceConfig):
            cleaned_lines = process_gh_dataset(config)

        cleaned_corpus_data[corpus_name] = cleaned_lines

    tqdm.write('Truncating Corpora...')
    corpus_splits = _compute_corpus_splits(cleaned_corpus_data, CORPUS_CONFIG, EXPERIMENT_CONFIG.random_seed)

    for split_name, lines in corpus_splits.items():
        config = get_dataset_config(split_name, CORPUS_CONFIG)
        save_jsonl(
                lines,
                PROCESSED_DATA_DIR,
                split_name,
                config.min_len_chars,
                config.min_len_words,
                config.max_len_words,
            )
    tqdm.write('Data Processing Complete')
