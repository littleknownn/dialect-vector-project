from src.config import (
    DatasetConfig,
    CoraalConfig,
    HFDatasetConfig,
    GithubDatasetConfig,
    corpus_config,
)
from src.utils import save_jsonl, processed_data_dir
from datasets import load_dataset
import requests
import re
from tqdm import tqdm
from typing import Optional


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


def process_coraal_dataset(config: CoraalConfig) -> list[str]:
    cleaned_lines = []
    for city_dir in tqdm(config.in_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for text_path in list(city_dir.glob('*.txt')):
            if text_path.name.startswith('._'):
                continue
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                lines = text_file.readlines()
                lines = lines[1:] if config.header else lines
                cleaned_lines += _process_coraal_lines(lines, config)
    return cleaned_lines


def process_hf_dataset(config: HFDatasetConfig) -> list[str]:
    dataset = load_dataset(config.hf_path, split=config.split)
    cleaned_lines = [clean_row
              for row in tqdm(dataset)
              if (clean_row := _clean_text(row[config.text_col], config))]
    return cleaned_lines


def process_gh_dataset(config: GithubDatasetConfig) -> list[str]:
    lyrics = requests.get(config.git_url)
    lyrics.raise_for_status()
    lines = lyrics.text.split(config.delimiter)
    cleaned_lines = [cleaned_line
                    for line in tqdm(lines)
                    if (cleaned_line := _clean_text(line, config))]
    return cleaned_lines


if __name__ == '__main__':
    for corpus_name, config in corpus_config.items():
        tqdm.write(f'Processing {corpus_name} Data...')

        if isinstance(config, CoraalConfig):
            cleaned_lines = process_coraal_dataset(config)

        elif isinstance(config, HFDatasetConfig):
            cleaned_lines = process_hf_dataset(config)

        elif isinstance(config, GithubDatasetConfig):
            cleaned_lines = process_gh_dataset(config)
        
        save_jsonl(
            cleaned_lines,
            processed_data_dir,
            corpus_name,
            config.min_len_chars,
            config.min_len_words
        )
    tqdm.write('Data Processing Complete')
