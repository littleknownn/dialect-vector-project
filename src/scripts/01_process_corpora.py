from src.utils import processed_data_dir
from src.config import (
    ExperimentConfig,
    DatasetConfig,
    CoraalConfig,
    HFDatasetConfig,
    GithubDatasetConfig,
    experiment_config,
    corpus_config,
)
from pathlib import Path
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from typing import Optional, Iterable
import requests


def _save_jsonl(
        text_iter: Iterable[str],
        out_dir: Path,
        out_filename: str,
        min_len_chars: int,
        min_len_words: int
    ) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            if len(text) < min_len_chars or len(text.split()) < min_len_words:
                continue
            json_object = {'text': text}
            out_file.write(json.dumps(json_object) + '\n')


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


def _process_coraal_lines(lines: list[str], config: CoraalConfig) -> list[str]:
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
        cols = line.split(config.delimiter)
        if len(cols) <= config.text_col_idx:
            continue
        speaker = cols[config.speaker_col_idx]
        text = cols[config.text_col_idx]

        if config.interviewer_marker in speaker:
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


def process_coraal_dataset(experiment_config: ExperimentConfig, config: CoraalConfig, out_filename: str) -> None:
    cleaned_lines = []
    for city_dir in tqdm(config.in_dir.iterdir(), desc=f'Processing {out_filename} Data'):
        if not city_dir.is_dir():
            continue
        for text_path in list(city_dir.glob('*.txt')):
            if text_path.name.startswith('._'):
                continue
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                lines = text_file.readlines()
                lines = lines[1:] if config.header else lines
                cleaned_lines += _process_coraal_lines(lines, config)
    _save_jsonl(
        cleaned_lines,
        processed_data_dir,
        out_filename,
        config.min_len_chars,
        config.min_len_words
    )


def process_hf_dataset(experiment_config: ExperimentConfig, config: HFDatasetConfig, out_filename: str) -> None:
    dataset = load_dataset(config.hf_path, split=config.split)
    cleaned_lines = [clean_row
              for row in tqdm(dataset, desc=f'Processing {out_filename} Data')
              if (clean_row := _clean_text(row[config.text_col], config))]
    _save_jsonl(
        cleaned_lines,
        processed_data_dir,
        out_filename,
        config.min_len_chars,
        config.min_len_words
    )


def process_gh_dataset(experiment_config: ExperimentConfig, config: GithubDatasetConfig, out_filename: str) -> None:
    lyrics = requests.get(config.git_url)
    lyrics.raise_for_status()
    lines = lyrics.text.split(config.delimiter)
    cleaned_lines = [cleaned_line
                    for line in tqdm(lines, desc=f'Processing {out_filename} Data')
                    if (cleaned_line := _clean_text(line, config))]
    _save_jsonl(
        cleaned_lines,
        processed_data_dir,
        out_filename,
        config.min_len_chars,
        config.min_len_words
    )

if __name__ == '__main__':
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    for corpus_name, config in corpus_config.items():
        if isinstance(config, CoraalConfig):
            process_coraal_dataset(experiment_config, config, corpus_name)

        if isinstance(config, HFDatasetConfig):
            process_hf_dataset(experiment_config, config, corpus_name)

        if isinstance(config, GithubDatasetConfig):
            process_gh_dataset(experiment_config, config, corpus_name)
    print('Data Processing Complete')