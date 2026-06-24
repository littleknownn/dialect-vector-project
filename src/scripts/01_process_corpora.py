from pathlib import Path
from src.utils import raw_data_dir, processed_data_dir
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple


@dataclass
class CoraalConfig:
    in_dir: Path
    header: bool
    delimiter: str
    text_col_idx: int
    speaker_col_idx: int
    interviewer_marker: str
    skipped_data_patterns: List[str]
    skipped_case_patterns: List[str]
    replacement_patterns: List[Tuple[str, str]]


@dataclass
class HFDatasetConfig:
    hf_path: str
    split: str
    text_col: str
    out_filename: str


def save_jsonl(text_iter: Iterable[str], out_dir: Path, out_filename: str, min_len: int=10) -> None:
    out_filename += '.jsonl'
    out_path = out_dir / out_filename
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for text in text_iter:
            if len(text) < min_len:
                continue
            json_object = {'text': text}
            out_file.write(json.dumps(json_object) + '\n')


def clean_text(text: str, config: CoraalConfig) -> Optional[str]:
    text_lowercase = text.lower()
    for pattern in config.skipped_data_patterns:
        if re.search(pattern, text_lowercase):
            return None    
    
    for pattern, replacement in config.replacement_patterns:
        text = re.sub(pattern, replacement, text)
    cleaned_text = text.strip()
    
    for pattern in config.skipped_case_patterns:
        if re.search(pattern, cleaned_text):
            return None
        
    if not cleaned_text:
        return None
    return cleaned_text


def _process_coraal_lines(lines: List[str], config: CoraalConfig) -> List[str]:
    utterances = []
    current_utterance = []
    current_speaker = None

    def flush_utterance():
        if current_utterance:
            combined_utterance = ' '.join(current_utterance)
            clean_utterance = clean_text(combined_utterance, config)
            if clean_utterance:
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


def process_coraal(config: CoraalConfig) -> None:
    coraal_dir = Path(processed_data_dir / 'CORAAL')
    coraal_dir.mkdir(parents=True, exist_ok=True)
    for city_dir in config.in_dir.iterdir():
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        city_text = []

        for text_path in tqdm(list(city_dir.glob('*.txt')), desc=f'Processing CORAAL {city} Data'):
            with open(text_path, 'r', encoding='utf-8') as text_file:
                lines = text_file.readlines()
                lines = lines[1:] if config.header else lines
                city_text += _process_coraal_lines(lines, config)

        if city_text:
            save_jsonl(city_text, coraal_dir, f'coraal_{city}_clean')


def process_hf_dataset(config: HFDatasetConfig) -> None:
    dataset = load_dataset(config.hf_path, split=config.split)
    tweets = [row[config.text_col].strip() for row in tqdm(dataset, desc='Processing TwitterAAE Data') if row[config.text_col]]
    if tweets:
        save_jsonl(tweets, processed_data_dir, config.out_filename)


coraal_config = CoraalConfig(
    in_dir = raw_data_dir / 'coraal',

    header = True,
    delimiter = '\t',
    text_col_idx = 3,
    speaker_col_idx = 1,
    interviewer_marker = '_int_',

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
    out_filename = 'TwitterAAE',
)


if __name__ == '__main__':
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    process_coraal(coraal_config)
    process_hf_dataset(twitteraae_config)
    print('Data Processing Complete')