from pathlib import Path
from typing import NamedTuple

from dataclasses import dataclass

from src.core.utils import RAW_DATA_DIR


@dataclass
class ExperimentConfig:
    model_hf_path: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str

    target_layers: list[int]
    target_ranks: list[int]
    batch_size: int
    max_seq_len: int
    random_seed: int


@dataclass
class DatasetConfig:
    min_len_chars: int
    min_len_words: int
    max_len_words: int

    skipped_data_patterns: list[str]
    skipped_case_patterns: list[str]
    replacement_patterns: list[tuple[str, str]]


@dataclass
class SupervisedDatasetConfig(DatasetConfig):
    aae_split_suffix: str
    sae_split_suffix: str

    supervised_split_suffix: str
    testing_split_suffix: str


@dataclass
class RealAAEConfig(SupervisedDatasetConfig):
    in_dir: Path
    header: bool
    delimiter: str
    
    aae_col_idx: int
    sae_col_idx: int


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


class SupervisedPair(NamedTuple):
    corpus_name: str
    aae_split: str
    sae_split: str


class EstimatorData(NamedTuple):
    supervised_split_pairs: list[SupervisedPair]
    aae_corpora_splits: list[str]


def get_estimator_data(corpus_config: dict[str, DatasetConfig]) -> EstimatorData:
    supervised_split_pairs = []
    aae_corpora_splits = []

    for corpus_name, config in corpus_config.items():
        if isinstance(config, SupervisedDatasetConfig):
            supervised_split_pairs.append(
                SupervisedPair(
                    corpus_name = corpus_name,
                    aae_split = corpus_name + config.aae_split_suffix + config.supervised_split_suffix,
                    sae_split = corpus_name + config.sae_split_suffix + config.supervised_split_suffix,
                )
            )
        else:
            aae_corpora_splits.append(corpus_name)

    estimator_data = EstimatorData(
        supervised_split_pairs,
        aae_corpora_splits,
    )
    return estimator_data


EXPERIMENT_CONFIG = ExperimentConfig(
    model_hf_path = 'meta-llama/Llama-3.2-3B',
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',

    target_layers = [15, 20],
    target_ranks = [2, 4, 8],
    batch_size = 2,
    max_seq_len = 512,
    random_seed = 42,
)


CORPUS_CONFIG = {
    'coraal_merged': CoraalConfig(
        in_dir = RAW_DATA_DIR / 'coraal',
        header = True,
        delimiter = '\t',
        text_col_idx = 3,
        speaker_col_idx = 1,
        interviewer_marker = '_int_',

        min_len_chars = 15,
        min_len_words = 5,
        max_len_words = 400,

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

    'real_aae': RealAAEConfig(
        in_dir = RAW_DATA_DIR / 'real_aae',
        header = True,
        delimiter = ',',
        aae_col_idx = 0,
        sae_col_idx = 1,
        aae_split_suffix = '_aae',
        sae_split_suffix = '_sae',
        supervised_split_suffix = '_supervised',
        testing_split_suffix = '_testing',

        min_len_chars = 15,
        min_len_words = 5,
        max_len_words = 400,

        skipped_data_patterns = [],
        skipped_case_patterns = [],
        replacement_patterns = [
            (r'(?:http|www\.)\S+', ''),
            (r'^RT\s+', ''),
            (r'&amp;|&lt;|&gt;', ''),
            (r'@\w+', '<Name>'),
            (r'\s+', ' '),
        ],
    ),

    'aave_lyrics': GithubDatasetConfig(
        git_url = 'https://raw.githubusercontent.com/jazmiahenry/aave_corpora/refs/heads/main/corpora/lyrics_corpora.txt',
        delimiter = '\n',

        min_len_chars = 15,
        min_len_words = 5,
        max_len_words = 400,

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
        max_len_words = 400,

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


MODEL_LAYERS_MAP = {
    'qwen2': 'model.layers.{}',
    'llama': 'model.layers.{}',
}
