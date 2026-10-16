import csv
import itertools
from pathlib import Path
import re
from typing import NamedTuple

from datasets import load_dataset
import requests
import spacy
from tqdm import tqdm
import torch

from src.core.config import (
    CORPUS_CONFIG,
    EXPERIMENT_CONFIG,
    CoraalSourceConfig,
    DatasetConfig,
    ExperimentConfig,
    GithubSourceConfig,
    HFSourceConfig,
    RealAaeSourceConfig,
    SupervisedDatasetConfig,
    get_dataset_config,
)
from src.core.paths import (
    PROCESSED_DATA_DIR,
    PROCESSING_LOG_FILENAME,
)
import src.core.text_utils as text_utils
import src.core.text_rules as text_rules
from src.core.utils import (
    LinesProcessingCounts,
    PairedCorpusProcessingCounts,
    PairMetadata,
    ProcessingLog,
    SplitSuffixes,
    save_json,
    save_jsonl,
)


class PairedCorpusData(NamedTuple):
    aae_lines: list[str]
    sae_lines: list[str]
    aae_back_lines: list[str]
    pair_metadata: list[PairMetadata]
    num_rows: int


class ProcessedCorpora(NamedTuple):
    splits: dict[str, list]
    processing_log: ProcessingLog


class DroppedSplitsSummary(NamedTuple):
    splits_after_drop: dict[str, list[str | PairMetadata]]
    num_dropped_by_split: dict[str, int]


def _process_coraal_lines(lines: list[str], config: DatasetConfig) -> list[str]:
    utterances = []
    current_utterance = []
    current_speaker = None

    def flush_utterance():
        if current_utterance:
            combined_utterance = ' '.join(current_utterance)
            if (clean_utterance := text_utils.clean_corpus_text(combined_utterance, config)):
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


def _process_coraal_dataset(config: DatasetConfig) -> list[str]:
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


def _process_realaae_dataset(config: SupervisedDatasetConfig) -> PairedCorpusData:
    aae_lines, sae_lines, aae_back_lines, pair_metadata = [], [], [], []
    dataset_filepath = next(config.source_config.in_dir.glob('*.csv'))
    with open(dataset_filepath, 'r', encoding='utf-8', errors='ignore') as dataset_file:
        csv_reader = csv.reader(dataset_file, delimiter=config.source_config.delimiter)
        if config.source_config.header:
            next(csv_reader)
        rows = list(csv_reader)
    
    for row_idx, row in enumerate(rows):
        aae_row = row[config.source_config.aae_col_idx]
        sae_row = row[config.source_config.sae_col_idx]
        if not (
            (clean_aae_row := text_utils.clean_corpus_text(aae_row, config))
            and (clean_sae_row := text_utils.clean_corpus_text(sae_row, config))
            and text_utils.check_text_len_bounds(clean_aae_row, config.min_len_chars, config.min_len_words, config.max_len_words)
            and text_utils.check_text_len_bounds(clean_sae_row, config.min_len_chars, config.min_len_words, config.max_len_words)
        ):
            continue
        aae_back_row = row[config.source_config.aae_back_col_idx]
        clean_aae_back_row = text_utils.clean_corpus_text(aae_back_row, config) or ''

        aae_lines.append(clean_aae_row)
        sae_lines.append(clean_sae_row)
        aae_back_lines.append(clean_aae_back_row)

        is_sae_typography_normalized = sae_row != sae_row.translate(text_rules.CLEANUP_REPLACEMENT_MAP)
        is_aae_back_typography_normalized = aae_back_row != aae_back_row.translate(text_rules.CLEANUP_REPLACEMENT_MAP)
        is_aae_back_valid = text_utils.check_text_len_bounds(clean_aae_back_row, config.min_len_chars, config.min_len_words, config.max_len_words)
        
        pair_metadata.append(PairMetadata(
            row_idx=row_idx,
            is_sae_typography_normalized=is_sae_typography_normalized,
            is_aae_back_typography_normalized=is_aae_back_typography_normalized,
            is_aae_back_valid=is_aae_back_valid,
            boundary=text_utils.compute_pair_boundary(clean_aae_row, clean_sae_row, clean_aae_back_row, config),
        ))

    paired_corpus_data = PairedCorpusData(
        aae_lines,
        sae_lines,
        aae_back_lines,
        pair_metadata,
        num_rows=len(rows),
    )
    return paired_corpus_data

def _process_hf_dataset(
        config: DatasetConfig,
) -> list[str]:
    dataset = load_dataset(config.source_config.hf_path, name=config.source_config.hf_name, split=config.source_config.split)
    cleaned_lines = []

    if config.sentencize:
        sentencizer = spacy.blank('en')
        sentencizer.add_pipe('sentencizer')

    for row in tqdm(dataset):
        text = row[config.source_config.text_col]
        if clean_text := text_utils.clean_corpus_text(text, config):
            if config.sentencize:
                cleaned_lines.extend([sent.text for sent in sentencizer(clean_text).sents])
            else:
                cleaned_lines.append(clean_text)
    return cleaned_lines


def _process_gh_dataset(config: DatasetConfig) -> list[str]:
    lyrics = requests.get(config.source_config.git_url)
    lyrics.raise_for_status()
    lines = lyrics.text.split(config.source_config.delimiter)
    cleaned_lines = [
        cleaned_line
        for line in tqdm(lines)
        if (cleaned_line := text_utils.clean_corpus_text(line, config))
    ]
    return cleaned_lines


def _compute_corpus_splits(
    corpus_data: dict[str, list[str | PairMetadata]],
    corpus_config: dict[str, DatasetConfig],
    split_size: int,
    seed: int,
) -> dict[str, list[str | PairMetadata]]:
    corpus_splits = {}
    permuted_indices_by_dataset = {}
    for split_name, lines in corpus_data.items():
        config = get_dataset_config(split_name, corpus_config)
        is_supervised = isinstance(config, SupervisedDatasetConfig)
        min_num_lines = 2 * split_size if is_supervised else split_size
        num_lines = len(lines)
        if num_lines < min_num_lines:
            raise ValueError(f'{split_name} has {num_lines} lines but needs minimum {min_num_lines}')

        config_id = id(config)
        if config_id not in permuted_indices_by_dataset:
            generator = torch.Generator().manual_seed(seed)
            permuted_indices_by_dataset[config_id] = torch.randperm(num_lines, generator=generator).tolist()
        permuted_lines = [lines[i] for i in permuted_indices_by_dataset[config_id]]

        if is_supervised:
            corpus_splits[f'{split_name}{SplitSuffixes.SUPERVISED}'] = permuted_lines[:split_size]
            corpus_splits[f'{split_name}{SplitSuffixes.TESTING}'] = permuted_lines[split_size:2 * split_size]
            corpus_splits[f'{split_name}{SplitSuffixes.RESERVE}'] = permuted_lines[2 * split_size:]
        else:
            corpus_splits[split_name] = permuted_lines[:split_size]
            if config.pool_size is not None:
                corpus_splits[f'{split_name}{SplitSuffixes.POOL}'] = permuted_lines[:config.pool_size]
        
    return corpus_splits


def _compute_stripped_splits(
    corpus_splits: dict[str, list[str | PairMetadata]],
    corpus_name: str,
) -> dict[str, list[str]]:
    stripped_splits = {}
    for dialect_suffix, split_suffix in itertools.product(
        (SplitSuffixes.AAE, SplitSuffixes.SAE),
        (SplitSuffixes.SUPERVISED, SplitSuffixes.TESTING),
    ):
        split_name = f'{corpus_name}{dialect_suffix}{split_suffix}'
        stripped_splits[f'{split_name}{SplitSuffixes.STRIPPED}'] = [
            text_rules.TERMINAL_PUNCTUATION_PATTERN.sub('', line.strip())
            for line in corpus_splits[split_name]
        ]
    return stripped_splits


def _drop_sae_artifact_pairs(
    corpus_splits: dict[str, list[str | PairMetadata]],
    corpus_name: str,
) -> DroppedSplitsSummary:
    paired_column_suffixes = (
        SplitSuffixes.AAE,
        SplitSuffixes.SAE,
        SplitSuffixes.AAE_BACK,
        SplitSuffixes.PAIR_METADATA,
    )
    splits_after_drop = {}
    num_dropped_by_split = {}
    for split_suffix in (SplitSuffixes.SUPERVISED, SplitSuffixes.TESTING, SplitSuffixes.RESERVE):
        sae_lines = corpus_splits[f'{corpus_name}{SplitSuffixes.SAE}{split_suffix}']
        kept_indices = [
            i for i, line in enumerate(sae_lines)
            if not text_rules.SAE_ARTIFACT_PATTERN.search(line)
        ]
        for col_suffix in paired_column_suffixes:
            split_name = f'{corpus_name}{col_suffix}{split_suffix}'
            splits_after_drop[split_name] = [corpus_splits[split_name][i] for i in kept_indices]
        num_dropped_by_split[split_suffix] = len(sae_lines) - len(kept_indices)

    dropped_splits_summary = DroppedSplitsSummary(
        splits_after_drop,
        num_dropped_by_split,
    )
    return dropped_splits_summary


def load_and_process_corpora(
    corpus_config: dict[str, DatasetConfig],
    experiment_config: ExperimentConfig,
) -> ProcessedCorpora:
    paired_corpus_config = {
        corpus_name: config for corpus_name, config in corpus_config.items()
        if isinstance(config, SupervisedDatasetConfig)
    }
    corpus_data = {}
    paired_corpus_counts, single_corpus_counts = {}, {}
    seen_lines = set()

    for corpus_name, config in paired_corpus_config.items():
        tqdm.write(f'Processing {corpus_name} Data...')
        if not isinstance(config.source_config, RealAaeSourceConfig):
            raise ValueError(f'Configure {corpus_name} SourceConfig object')
        paired_corpus_data = _process_realaae_dataset(config)
        seen_lines |= {line.lower() for line in paired_corpus_data.aae_lines}

        corpus_data[f'{corpus_name}{SplitSuffixes.AAE}'] = paired_corpus_data.aae_lines
        corpus_data[f'{corpus_name}{SplitSuffixes.SAE}'] = paired_corpus_data.sae_lines
        corpus_data[f'{corpus_name}{SplitSuffixes.AAE_BACK}'] = paired_corpus_data.aae_back_lines
        corpus_data[f'{corpus_name}{SplitSuffixes.PAIR_METADATA}'] = paired_corpus_data.pair_metadata

        num_aae_back_invalid = sum(not metadata['is_aae_back_valid'] for metadata in paired_corpus_data.pair_metadata)
        paired_corpus_counts[corpus_name] = PairedCorpusProcessingCounts(
            num_rows=paired_corpus_data.num_rows,
            num_pairs_kept=len(paired_corpus_data.aae_lines),
            num_aae_back_invalid=num_aae_back_invalid,
            num_dropped_by_split={}
        )

    for corpus_name, config in corpus_config.items():
        if corpus_name in paired_corpus_config:
            continue
        tqdm.write(f'Processing {corpus_name} Data...')

        if isinstance(config.source_config, CoraalSourceConfig):
            cleaned_lines = _process_coraal_dataset(config)
        elif isinstance(config.source_config, HFSourceConfig):
            cleaned_lines = _process_hf_dataset(config)
        elif isinstance(config.source_config, GithubSourceConfig):
            cleaned_lines = _process_gh_dataset(config)

        lines = []
        num_out_of_bounds, num_duplicates = 0, 0
        for line in cleaned_lines:
            if not text_utils.check_text_len_bounds(line, config.min_len_chars, config.min_len_words, config.max_len_words):
                num_out_of_bounds += 1
                continue
            if (line_lower := line.lower()) in seen_lines:
                num_duplicates += 1
                continue
            lines.append(line)
            seen_lines.add(line_lower)

        corpus_data[corpus_name] = lines
        single_corpus_counts[corpus_name] = LinesProcessingCounts(
            num_cleaned=len(cleaned_lines),
            num_out_of_bounds=num_out_of_bounds,
            num_duplicates=num_duplicates,
            num_kept=len(lines)
        )

    tqdm.write('Splitting Corpora...')
    corpus_splits = _compute_corpus_splits(corpus_data, corpus_config, experiment_config.corpus_split_size, experiment_config.random_seed)
    for corpus_name in paired_corpus_config:
        dropped_splits_summary = _drop_sae_artifact_pairs(corpus_splits, corpus_name)
        corpus_splits |= dropped_splits_summary.splits_after_drop
        paired_corpus_counts[corpus_name]['num_dropped_by_split'] = dropped_splits_summary.num_dropped_by_split
        corpus_splits |= _compute_stripped_splits(corpus_splits, corpus_name)

    split_sizes = {split_name: len(lines) for split_name, lines in corpus_splits.items()}
    processing_log = ProcessingLog(
        paired_corpus_counts=paired_corpus_counts,
        single_corpus_counts=single_corpus_counts,
        split_sizes=split_sizes,
    )
    processed_corpora = ProcessedCorpora(corpus_splits, processing_log)
    return processed_corpora


def extract_feature_indices_dict(
    supervised_corpus_name: str,
    corpus_splits: dict[str, list[str]],
    feature_patterns_dict: dict[str, re.Pattern] = text_rules.AAE_FEATURE_PATTERNS,
    unmatched_feature_key: str = text_rules.UNMATCHED_FEATURE_KEY,
) -> dict[str, dict[str, str]]:
    testing_split_key = f'{supervised_corpus_name}{SplitSuffixes.AAE}{SplitSuffixes.TESTING}'
    feature_indices_dict = {feature: {} for feature in feature_patterns_dict}
    feature_indices_dict[unmatched_feature_key] = {}
    
    for i, line in enumerate(corpus_splits[testing_split_key]):
        matched_flag = False
        for feature, pattern in feature_patterns_dict.items():
            if pattern.search(line.lower()):
                feature_indices_dict[feature][str(i)] = line
                matched_flag = True
        if not matched_flag:
            feature_indices_dict[unmatched_feature_key][str(i)] = line
    return feature_indices_dict


def save_processed_corpora(
    processed_corpora: ProcessedCorpora,
    corpus_config: dict[str, DatasetConfig],
    feature_patterns_dict: dict[str, re.Pattern] = text_rules.AAE_FEATURE_PATTERNS,
    unmatched_feature_key: str = text_rules.UNMATCHED_FEATURE_KEY,
    processing_log_filename: str = PROCESSING_LOG_FILENAME,
    out_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    for split_name, lines in processed_corpora.splits.items():
        if SplitSuffixes.PAIR_METADATA in split_name:
            save_json(lines, out_dir, split_name)
        else:
            save_jsonl(lines, out_dir, split_name)
    for corpus_name, config in corpus_config.items():
        if isinstance(config, SupervisedDatasetConfig):
            feature_indices_dict = extract_feature_indices_dict(corpus_name, processed_corpora.splits, feature_patterns_dict=feature_patterns_dict, unmatched_feature_key=unmatched_feature_key)
            save_json(feature_indices_dict, out_dir, f'{corpus_name}{SplitSuffixes.FEATURE_INDICES}')
    save_json(processed_corpora.processing_log, out_dir, processing_log_filename)


if __name__ == '__main__':
    processed_corpora = load_and_process_corpora(CORPUS_CONFIG, EXPERIMENT_CONFIG)
    save_processed_corpora(processed_corpora, CORPUS_CONFIG)
    tqdm.write('01: Corpus Processing Complete\n')

