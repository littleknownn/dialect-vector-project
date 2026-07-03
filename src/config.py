from src.utils import raw_data_dir
from dataclasses import dataclass
from pathlib import Path
from typing import Union

@dataclass
class ExperimentConfig:
    model_hf_path: str
    lora_rank: int
    lora_alpha: int
    target_weight_matrices: Union[list[str], str]

    max_training_samples: int
    batch_size: int
    max_seq_len: int
    learning_rate: float
    epochs: int
    random_seed: int


@dataclass
class DatasetConfig:
    min_len_chars: int
    min_len_words: int

    skipped_data_patterns: list[str]
    skipped_case_patterns: list[str]
    replacement_patterns: list[tuple[str, str]]


@dataclass
class CoraalConfig(DatasetConfig):
    in_dir: Path
    header: bool
    delimiter: str
    text_col_idx: int
    speaker_col_idx: int
    interviewer_marker: str


@dataclass
class HFDatasetConfig(DatasetConfig):
    hf_path: str
    split: str
    text_col: str


@dataclass
class GithubDatasetConfig(DatasetConfig):
    git_url: str
    delimiter: str


experiment_config = ExperimentConfig(
    model_hf_path='Qwen/Qwen2.5-1.5B',
    lora_rank = 8,
    lora_alpha = 16,
    target_weight_matrices = ['v_proj', 'o_proj'], # ['q_proj', 'v_proj'], 'all-linear'

    max_training_samples = 5499,
    batch_size = 2,
    max_seq_len = 512,
    learning_rate = 2e-4,
    epochs = 1,
    random_seed = 42,
)


corpus_config = {
    'coraal_merged': CoraalConfig(
        in_dir = raw_data_dir / 'coraal',
        header = True,
        delimiter = '\t',
        text_col_idx = 3,
        speaker_col_idx = 1,
        interviewer_marker = '_int_',

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
    ),

    'aave_lyrics': GithubDatasetConfig(
        git_url = 'https://raw.githubusercontent.com/jazmiahenry/aave_corpora/refs/heads/main/corpora/lyrics_corpora.txt',
        delimiter = '\n',

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
    ),

    'twitter_aae': HFDatasetConfig(
        hf_path = 'lighteval/TwitterAAE',
        split = 'test',
        text_col = 'tweet',

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
}