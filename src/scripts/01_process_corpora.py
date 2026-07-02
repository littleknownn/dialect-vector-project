from src.utils import raw_data_dir, processed_data_dir
from src.config import ExperimentConfig, experiment_config
from pathlib import Path
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple
import requests


@dataclass
class DatasetConfig:
    min_len_chars: int
    min_len_words: int

    skipped_data_patterns: List[str]
    skipped_case_patterns: List[str]
    replacement_patterns: List[Tuple[str, str]]


@dataclass
class CoraalConfig(DatasetConfig):
    in_dir: Path
    header: bool
    delimiter: str
    text_col_idx: int
    speaker_col_idx: int
    interviewer_marker: str
    out_filename: str


@dataclass
class HFDatasetConfig(DatasetConfig):
    hf_path: str
    split: str
    text_col: str
    out_filename: str


@dataclass
class GithubDatasetConfig(DatasetConfig):
    git_url: str
    delimiter: str
    out_filename: str
    

def save_jsonl(
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


def clean_text(text: str, config: DatasetConfig) -> Optional[str]:
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


def _process_coraal_lines(lines: List[str], config: CoraalConfig) -> List[str]:
    utterances = []
    current_utterance = []
    current_speaker = None

    def flush_utterance():
        if current_utterance:
            combined_utterance = ' '.join(current_utterance)
            if (clean_utterance := clean_text(combined_utterance, config)):
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


def process_coraal_dataset(experiment_config: ExperimentConfig, config: CoraalConfig) -> None:
    cleaned_lines = []
    for city_dir in tqdm(config.in_dir.iterdir(), desc=f'Processing CORAAL Data'):
        if not city_dir.is_dir():
            continue
        for text_path in list(city_dir.glob('*.txt')):
            if text_path.name.startswith('._'):
                continue
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                lines = text_file.readlines()
                lines = lines[1:] if config.header else lines
                cleaned_lines += _process_coraal_lines(lines, config)
    save_jsonl(
        cleaned_lines,
        processed_data_dir,
        config.out_filename,
        config.min_len_chars,
        config.min_len_words
    )


def process_hf_dataset(experiment_config: ExperimentConfig, config: HFDatasetConfig) -> None:
    dataset = load_dataset(config.hf_path, split=config.split)
    cleaned_lines = [clean_row
              for row in tqdm(dataset, desc='Processing TwitterAAE Data')
              if (clean_row := clean_text(row[config.text_col], config))]
    save_jsonl(
        cleaned_lines,
        processed_data_dir,
        config.out_filename,
        config.min_len_chars,
        config.min_len_words
    )


def process_gh_dataset(experiment_config: ExperimentConfig, config: GithubDatasetConfig) -> None:
    lyrics = requests.get(config.git_url)
    lyrics.raise_for_status()
    lines = lyrics.text.split(config.delimiter)
    cleaned_lines = [cleaned_line
                    for line in tqdm(lines, desc='Processing AAVELyrics Data')
                    if (cleaned_line := clean_text(line, config))]
    save_jsonl(
        cleaned_lines,
        processed_data_dir,
        config.out_filename,
        config.min_len_chars,
        config.min_len_words
    )

if __name__ == '__main__':
    coraal_config = CoraalConfig(
        in_dir = raw_data_dir / 'coraal',
        header = True,
        delimiter = '\t',
        text_col_idx = 3,
        speaker_col_idx = 1,
        interviewer_marker = '_int_',

        out_filename = 'coraal_merged',
        min_len_chars = 15,
        min_len_words = 5,

        skipped_data_patterns = [
            r'rd-',
            r'\(pause \d+\.\d+\)',
            r'inaudible',
            r'unintelligible',
            r'\?\?+',
            r'(^|[\[\]\(\)<>])\s*\?',
        ],
        skipped_case_patterns = [
            r'(|\s)\?(\s|$)',
            r'-\s*[.,!?]*\s*$',
            r'\b\w+-\s',
        ],
        replacement_patterns = [
            (r'\([^\)]+\)', ''),
            (r'<[^>]+>',    ''),
            (r'[\[\]/]',    ''),
            (r'\s+',        ' '),
        ],
    )

    twitteraae_config = HFDatasetConfig(
        hf_path = 'lighteval/TwitterAAE',
        split = 'test',
        text_col = 'tweet',
        out_filename = 'twitter_aae',

        min_len_chars = 15,
        min_len_words = 5,

        skipped_data_patterns = [],
        skipped_case_patterns = [],
        replacement_patterns = [
            (r'(?:http|www\.)\S+', ''),
            (r'^RT\s+', ''),
            (r'&amp;|&lt;|&gt;', ''),
            (r'@\w+', '<Name>'),
            (r'\s+', ' '),
        ],
    )

    lyrics_config = GithubDatasetConfig(
        git_url = 'https://raw.githubusercontent.com/jazmiahenry/aave_corpora/refs/heads/main/corpora/lyrics_corpora.txt',
        delimiter = '\n',
        out_filename = 'aave_lyrics',

        min_len_chars = 15,
        min_len_words = 5,

        skipped_data_patterns = [
            r'(?:^|\blyrics)(?:verse|chorus|intro|outro|hook|bridge|interlude|skit|narrator|scratches|produced|vocals sampled|beat changes)\b',
            r'\bsample[ds]?$',
        ],
        skipped_case_patterns = [
            r'^[a-z]+$',
        ],
        replacement_patterns = [
            (r'embed$', ''),
            (r'\s+', ' '),
        ],
    )

    processed_data_dir.mkdir(parents=True, exist_ok=True)
    process_coraal_dataset(experiment_config, coraal_config)
    process_hf_dataset(experiment_config, twitteraae_config)
    process_gh_dataset(experiment_config, lyrics_config)
    print('Data Processing Complete')